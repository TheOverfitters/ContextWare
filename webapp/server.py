import asyncio
import json
import os
import httpx
import re
from datetime import datetime
from pathlib import Path
from quart import Quart, render_template, request, make_response

import sys
from pathlib import Path

target_folder = Path(__file__).resolve().parent / '../acf-analysis'

sys.path.append(str(target_folder))

from acf_analyser import classify_diff, _check_context_budget

from acf_io import load_label_descriptions, load_categories, load_ollama_api_key
from acf_prompt import get_system_prompt, DEFAULT_CATEGORIES, DEFAULT_MAINTENANCE_TYPES, DEFAULT_LEAF_TO_CLASS, DEFAULT_LEAF_TO_BASE
from ollama_client import SlidingRateLimiter

app = Quart(__name__)

DEFAULT_MODEL = "gemma4:31b-cloud"

class MockArgs:
    diffs_json = Path("acf_diffs.json")
    label_descriptions = Path("categories.json")
    categories = None
    limit = None
    repo_filter = None
    resume = False
    include_patch = False
    include_chunk_details = False
    print_prompt = True

    # LLM Parameters and prompt engineering flags
    timestamp = None
    env_file = Path(".env")
    model = DEFAULT_MODEL
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    max_diff_chars = 90000
    max_tokens = 60000
    use_format_schema = True
    context_size = None
    retry_on_zero_flags = False
    no_prefill = False
    debug_raw_response = True
    temperature = 0.3
    top_p = 1.0
    timeout_seconds = 90
    retry_per_prompt = 3

    # Rate limiting parameters
    requests_per_minute = 600
    tokens_per_minute = 10000000
    min_request_interval_seconds = 0.0
    rate_limit_wait_base_seconds = 12
    rate_limit_wait_max_seconds = 45
    hard_cooldown_seconds = 90
    retry_on_invalid = 3

    # Chunking and aggregation parameters
    chunk_overlap_chars = 400
    chunk_aggregation = "max"
    per_category_threshold = 0.50
    diff_retry_on_missing_list = 3

args = MockArgs()

async def fetch_github_diff(repo_url: str):
    # Extract owner and repo from link, find the latest modification between AGENTS.md and CLAUDE.md
    # by comparing dates, and return the specific diff of that file
    # Robust URL parsing (handles trailing slashes)
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url.rstrip("/"))
    if not match:
        raise ValueError(f"Invalid Github URL: {repo_url}")
    
    owner, repo = match.groups()
    repo = repo.replace(".git", "")
    
    files_to_check = ["AGENTS.md", "CLAUDE.md"]
    
    # Prepare headers. Use token if present in environment variables (HIGHLY RECOMMENDED)
    headers = {"Accept": "application/vnd.github.v3+json"}
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    latest_commit_info = None

    async with httpx.AsyncClient(headers=headers) as client:
        # 1. Search for the latest commit for ALL files and compare dates
        for file_path in files_to_check:
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?path={file_path}&per_page=1"
            res = await client.get(commits_url)
            
            # Rate Limit handling
            if res.status_code in (403, 429):
                raise ConnectionError("GitHub rate limit exceeded. Set GITHUB_TOKEN environment variable")
            
            if res.status_code == 200:
                data = res.json()
                if data:  # If list is not empty, the file has a history
                    commit = data[0]
                    date_str = commit['commit']['author']['date']
                    # Convert ISO string (e.g. 2023-10-25T10:00:00Z) to datetime for comparison
                    commit_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    
                    # Save if it is the first one found, or if it is more recent than the one already saved
                    if not latest_commit_info or commit_date > latest_commit_info['date']:
                        latest_commit_info = {
                            "file": file_path,
                            "sha": commit['sha'],
                            "message": commit['commit']['message'],
                            "date": commit_date,
                            "date_str": date_str
                        }

        # If after checking all files nothing was found
        if not latest_commit_info:
            raise ValueError(f"No history found for {files_to_check} in repository {owner}/{repo}")

        # 2. Get the EXACT diff (patch) for the modified file
        # Request commit details in JSON format
        commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{latest_commit_info['sha']}"
        commit_res = await client.get(commit_url)
        
        if commit_res.status_code == 200:
            commit_data = commit_res.json()
            diff_text = ""
            
            # Search for the specific file among the modified files in the commit
            for file_info in commit_data.get('files', []):
                if file_info['filename'] == latest_commit_info['file']:
                    # .get('patch') contains the actual diff of the single file
                    diff_text = file_info.get('patch', "No textual diff available (file too large or renamed)")
                    break
            
            return {
                "diff_id": latest_commit_info['sha'][:7],
                "commit_hash": latest_commit_info['sha'],
                "commit_message": latest_commit_info['message'],
                "filename": latest_commit_info['file'],
                "diff_text": diff_text,
                "timestamp": latest_commit_info['date_str']
            }
        else:
            raise ValueError(f"Unable to retrieve commit details {latest_commit_info['sha']}")

def run_acf_analyser(diff_data):
    # Synchronous wrapper to execute the acf_analyser pipeline in a separate thread
    # Load categories and prompt
    label_categories, label_descriptions, label_examples = load_label_descriptions(Path("categories.json"))
    categories = load_categories(None, label_categories or DEFAULT_CATEGORIES)
    system_prompt = get_system_prompt(categories, label_descriptions, label_examples)
    
    rate_limiter = SlidingRateLimiter(600, 10_000_000, 0.0)
    
    # --- API KEY MANAGEMENT ---
    # Search for the .env file in the current folder or the parent folder
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path("../.env")
        
    api_key = load_ollama_api_key(env_path)
    
    # If .env file is not found, try to get it from system variables, 
    # otherwise leave it empty (which usually works for pure local Ollama)
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY", "")
    
    # Execute classification
    record = classify_diff(
        diff=diff_data,
        args=args, # assuming you have your args object
        categories=categories,
        maintenance_types=list(DEFAULT_MAINTENANCE_TYPES),
        leaf_to_class=dict(DEFAULT_LEAF_TO_CLASS),
        leaf_to_base=dict(DEFAULT_LEAF_TO_BASE),
        system_prompt=system_prompt,
        api_key=api_key,
        rate_limiter=rate_limiter
    )
    return record

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/stream')
async def stream():
    repo_url = request.args.get('repo', '')

    async def generate_events():
        try:
            # Phase 1: Diff search on Github
            yield f"event: status\ndata: {json.dumps({'message': 'Searching for AGENTS.md or CLAUDE.md on GitHub...'})}\n\n"
            diff_data = await fetch_github_diff(repo_url)
            
            # Phase 2: LLM Analysis
            msg = f"Diff found in {diff_data['filename']} (Commit: {diff_data['diff_id']}). LLM analysis in progress..."
            yield f"event: status\ndata: {json.dumps({'message': msg})}\n\n"
            
            # Execute blocking classification task in a separate thread
            record = await asyncio.to_thread(run_acf_analyser, diff_data)
            
            # Format results filtering score > 0
            scores = record.get('category_scores', {})
            flagged = record.get('flagged_categories', [])
            
            # Reconstruct output array by joining scores and rationale (which is inside flagged)
            results = []
            for cat, score in scores.items():
                if score > 0:
                    rationale = next((f['rationale'] for f in flagged if f['category'] == cat), "Rationale not available for this score")
                    results.append({"category": cat, "score": score, "rationale": rationale})
            
            # Sort by descending score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Send final data
            yield f"event: complete\ndata: {json.dumps({'results': results, 'filename': diff_data['filename']})}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc() # Print the real and complete error in your terminal
            yield f"event: stream_error\ndata: {json.dumps({'message': str(e)})}\n\n"

    response = await make_response(
        generate_events(),
        {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        },
    )
    response.timeout = None
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5000)