import asyncio
import asyncio.subprocess
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
from acf_io import load_dotenv_file
load_dotenv_file(Path("../.env"))
api_key = os.environ.get("OLLAMA_API_KEY", "")

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
    max_diff_chars = 100000
    max_tokens = 10000
    use_format_schema = True
    context_size = None
    retry_on_zero_flags = False
    no_prefill = True
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

async def fetch_github_diffs(repo_url: str, max_diffs: int = 50):
    # URL parsing
    match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url.rstrip("/"))
    if not match:
        raise ValueError(f"Invalid Github URL: {repo_url}")
    
    owner, repo = match.groups()
    repo = repo.replace(".git", "")
    
    files_to_check = ["AGENTS.md", "CLAUDE.md"]
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    all_commits_info = []

    async with httpx.AsyncClient(headers=headers) as client:
        # Retrieve the latest `max_diffs` commits for target files
        for file_path in files_to_check:
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits?path={file_path}&per_page={max_diffs}"
            res = await client.get(commits_url)
            
            # Rate Limit handling
            if res.status_code in (403, 429):
                raise ConnectionError("GitHub rate limit exceeded. Set GITHUB_TOKEN environment variable.")
            
            if res.status_code == 200:
                data = res.json()
                for commit in data:
                    date_str = commit['commit']['author']['date']
                    # Convert ISO string to datetime for comparison
                    commit_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    all_commits_info.append({
                        "file": file_path,
                        "sha": commit['sha'],
                        "message": commit['commit']['message'],
                        "date": commit_date,
                        "date_str": date_str
                    })

        # If after checking all files nothing was found
        if not all_commits_info:
            raise ValueError(f"No history found for {files_to_check} in repository {owner}/{repo}.")

        # Sort globally by date, most recent first
        all_commits_info.sort(key=lambda x: x['date'], reverse=True)
        
        # Truncate to max_diffs total
        all_commits_info = all_commits_info[:max_diffs]

        diffs_list = []
        # Get the EXACT diff (patch) for each selected commit
        for commit_info in all_commits_info:
            commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_info['sha']}"
            commit_res = await client.get(commit_url)
            
            if commit_res.status_code == 200:
                commit_data = commit_res.json()
                diff_text = ""
                
                for file_info in commit_data.get('files', []):
                    if file_info['filename'] == commit_info['file']:
                        diff_text = file_info.get('patch', "")
                        break
                
                # Add only if there is an actual textual diff
                if diff_text:
                    diffs_list.append({
                        "diff_id": commit_info['sha'][:7],
                        "commit_hash": commit_info['sha'],
                        "commit_message": commit_info['message'],
                        "filename": commit_info['file'],
                        "diff_text": diff_text,
                        "timestamp": commit_info['date_str']
                    })

    return diffs_list


def run_acf_analyser(diff_data):
    """Synchronous wrapper to execute the acf_analyser pipeline in a separate thread"""
    # Load categories and prompt
    label_categories, label_descriptions, label_examples = load_label_descriptions(Path("categories.json"))
    categories = load_categories(None, label_categories or DEFAULT_CATEGORIES)
    system_prompt = get_system_prompt(categories, label_descriptions, label_examples)
    
    rate_limiter = SlidingRateLimiter(600, 10_000_000, 0.0)
    
    # Search for the .env file in the current folder or the parent folder
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path("../.env")
        
    api_key = load_ollama_api_key(env_path)
    
    # If .env file is not found, try to get it from system variables, 
    # otherwise leave it empty
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY", "")
    
    # Execute classification
    record = classify_diff(
        diff=diff_data,
        args=args,
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
            yield f"event: status\ndata: {json.dumps({'message': 'Searching for AGENTS.md or CLAUDE.md history on GitHub...'})}\n\n"
            diffs_list = await fetch_github_diffs(repo_url, max_diffs=50)
            
            if not diffs_list:
                yield f"event: stream_error\ndata: {json.dumps({'message': 'No textual diffs found.'})}\n\n"
                return

            # Save the diffs to disk so run_multi_model.py can read them
            with open(args.diffs_json, "w", encoding="utf-8") as f:
                json.dump(diffs_list, f, indent=2)

            total_diffs = len(diffs_list)
            
            for idx, diff_data in enumerate(diffs_list):
                msg = f"Analysis {idx + 1}/{total_diffs} in progress (Commit: {diff_data['diff_id']})..."
                yield f"event: status\ndata: {json.dumps({'message': msg})}\n\n"
                
                record = await asyncio.to_thread(run_acf_analyser, diff_data)
                
                scores = record.get('category_scores', {})
                flagged = record.get('flagged_categories', [])
                
                results = []
                for cat, score in scores.items():
                    if score > 0:
                        rationale = next((f['rationale'] for f in flagged if f['category'] == cat), "Rationale not available for this score.")
                        results.append({"category": cat, "score": score, "rationale": rationale})
                
                results = sorted(results, key=lambda x: x['score'], reverse=True)
                
                payload = {
                    "diff_id": diff_data['diff_id'],
                    "filename": diff_data['filename'],
                    "commit_message": diff_data['commit_message'],
                    "timestamp": diff_data['timestamp'],
                    "results": results,
                    "maintenance": record.get("maintenance", {})
                }

                yield f"event: diff_result\ndata: {json.dumps(payload)}\n\n"
                
            yield f"event: complete\ndata: {json.dumps({'message': 'Analysis of the entire history completed!'})}\n\n"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
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


# Route for the multi-model script
@app.route('/stream_multi_model')
async def stream_multi_model():
    async def generate_multi_events():
        try:
            models_to_run = [
                "kimi-k2.7-code:cloud",
                "qwen3.5:cloud",
                "glm-5.2:cloud",
            ]
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            server_dir = Path(__file__).resolve().parent
            analyser_script = server_dir / "../acf-analysis/acf_analyser.py"
            agreement_script = server_dir / "../acf-analysis/agreement_analysis.py"
            output_dir = server_dir / "../acf-analysis/acf-outputs"
            
            env_path = server_dir / ".env"
            if not env_path.exists():
                env_path = server_dir.parent / ".env"
            
            from acf_io import load_dotenv_file
            load_dotenv_file(env_path)
            
            api_key = os.environ.get("OLLAMA_API_KEY", "")
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

            sub_env = os.environ.copy()
            sub_env["OLLAMA_API_KEY"] = api_key
            sub_env["OLLAMA_BASE_URL"] = base_url

            # Create an event queue for SSE stream
            queue = asyncio.Queue()
            
            # Put initial status in the queue
            await queue.put(f"event: multi_status\ndata: {json.dumps({'message': f'Starting multi-model validation for {len(models_to_run)} models...'})}\n\n")

            async def run_model(model):
                await queue.put(f"event: multi_status\ndata: {json.dumps({'message': f'[{model}] Booting instance...'})}\n\n")
                try:
                    cmd = [
                        sys.executable, "-u", str(analyser_script.resolve()),
                        "--model", model, "--timestamp", timestamp,
                        "--diffs-json", str(args.diffs_json.resolve()),
                        "--output-dir", str(output_dir.resolve()),
                        "--temperature", "0.0", "--env-file", str(env_path.resolve()),
                        "--ollama-base-url", base_url
                    ]
                    
                    process = await asyncio.create_subprocess_exec(
                        *cmd, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT, env=sub_env,
                        cwd=str(analyser_script.parent)
                    )
                    
                    while True:
                        line = await process.stdout.readline()
                        if not line: break
                        decoded_line = line.decode('utf-8', errors='replace').strip()
                        if decoded_line:
                            await queue.put(f"event: multi_status\ndata: {json.dumps({'message': f'[{model}] {decoded_line}'})}\n\n")
                            
                    await process.wait()
                    
                    if process.returncode != 0:
                        await queue.put(f"event: multi_status\ndata: {json.dumps({'message': f'[{model}] ERROR: Exited with code {process.returncode}'})}\n\n")
                    else:
                        await queue.put(f"event: multi_status\ndata: {json.dumps({'message': f'[{model}] DONE -> OK'})}\n\n")
                except Exception as e:
                    await queue.put(f"event: multi_status\ndata: {json.dumps({'message': f'[{model}] FAILED to launch: {str(e)}'})}\n\n")

            async def orchestrate():
                # Launch all models concurrently
                tasks = [asyncio.create_task(run_model(model)) for model in models_to_run]
                await asyncio.gather(*tasks)

                # --- AGREEMENT ANALYSIS PHASE ---
                await queue.put(f"event: multi_status\ndata: {json.dumps({'message': '[Analysis] Compiling inter-model agreement and ambiguity metrics...'})}\n\n")
                
                report_dir = Path(app.static_folder) / "reports" / timestamp if app.static_folder else Path("static/reports") / timestamp
                report_dir.mkdir(parents=True, exist_ok=True)
                
                cmd_agr = [
                    sys.executable, "-u", str(agreement_script.resolve()),
                    "--timestamp", timestamp, "--output-dir", str(output_dir.resolve()),
                    "--report-dir", str(report_dir.resolve()), "--models"
                ] + models_to_run
                
                try:
                    process_agr = await asyncio.create_subprocess_exec(
                        *cmd_agr, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.STDOUT, env=sub_env,
                        cwd=str(agreement_script.parent)
                    )

                    while True:
                        line = await process_agr.stdout.readline()
                        if not line: break
                        decoded_line = line.decode('utf-8', errors='replace').strip()
                        if decoded_line:
                            await queue.put(f"event: multi_status\ndata: {json.dumps({'message': f'[Analysis] {decoded_line}'})}\n\n")
                            
                    await process_agr.wait()

                    if process_agr.returncode == 0:
                        await queue.put(f"event: multi_complete\ndata: {json.dumps({'message': 'Validation complete!', 'report_id': timestamp})}\n\n")
                    else:
                        await queue.put(f"event: stream_error\ndata: {json.dumps({'message': f'Analysis failed with code {process_agr.returncode}'})}\n\n")
                except Exception as e:
                    await queue.put(f"event: stream_error\ndata: {json.dumps({'message': f'Agreement script failed: {str(e)}'})}\n\n")

                # Push sentinel to gracefully close the stream
                await queue.put(None)

            # Fire off the orchestrator in the background
            asyncio.create_task(orchestrate())

            # Consume the queue and yield to client
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield msg

        except Exception as e:
            yield f"event: stream_error\ndata: {json.dumps({'message': str(e)})}\n\n"

    response = await make_response(
        generate_multi_events(),
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