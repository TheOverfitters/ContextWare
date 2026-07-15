import asyncio
import asyncio.subprocess
import json
import os
import shutil
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
from acf_io import load_label_descriptions, load_categories, load_ollama_api_key, sanitize_model_name, load_maintenance_descriptions
from acf_prompt import get_system_prompt, DEFAULT_CATEGORIES, DEFAULT_MAINTENANCE_TYPES, DEFAULT_LEAF_TO_CLASS, DEFAULT_LEAF_TO_BASE
from ollama_client import SlidingRateLimiter
from acf_io import load_dotenv_file


_SERVER_DIR = Path(__file__).resolve().parent
for _env_candidate in (
    _SERVER_DIR / ".env",                       # webapp/.env
    _SERVER_DIR.parent / ".env",                # repo-root/.env
    _SERVER_DIR.parent / ".contextWare" / ".env",  # repo-root/.contextWare/.env
):
    load_dotenv_file(_env_candidate)

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
    temperature = 0.0
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


MEMORY_DIR = Path(__file__).resolve().parent / "memory"
STATIC_REPORTS_DIR = (
    Path(app.static_folder) / "reports"
    if app.static_folder
    else Path(__file__).resolve().parent / "static" / "reports"
)

# Create the cache roots at startup so they are visible immediately, even
# before the first analysis writes anything into them
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
STATIC_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    """Make a string usable as a file/folder name (Windows-safe: no ':')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "item"


def repo_slug(repo_url: str) -> str:
    """Turn a GitHub URL into a stable 'owner__repo' folder key."""
    match = re.search(r"github\.com/([^/]+)/([^/]+)", (repo_url or "").rstrip("/"))
    if not match:
        raise ValueError(f"Invalid Github URL: {repo_url}")
    owner, repo = match.groups()
    repo = repo.replace(".git", "")
    return _safe_name(f"{owner}__{repo}")


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


_CATEGORIES_PATH = _SERVER_DIR.parent / "categories.json"
_MR_PATH = _SERVER_DIR.parent / "modification_request.json"

_label_categories, _label_descriptions, _label_examples = load_label_descriptions(_CATEGORIES_PATH)
CATEGORIES = load_categories(None, _label_categories or DEFAULT_CATEGORIES)

_maint = load_maintenance_descriptions(_MR_PATH)
MAINTENANCE_TYPES = _maint["types"] or list(DEFAULT_MAINTENANCE_TYPES)
LEAF_TO_CLASS = _maint["leaf_to_class"] or dict(DEFAULT_LEAF_TO_CLASS)
LEAF_TO_BASE = _maint["leaf_to_base"] or dict(DEFAULT_LEAF_TO_BASE)

SYSTEM_PROMPT = get_system_prompt(
    CATEGORIES,
    _label_descriptions,
    _label_examples,
    maintenance_types=MAINTENANCE_TYPES,
    maintenance_descriptions=_maint["descriptions"],
    maintenance_examples=_maint["examples"],
    maintenance_classes=_maint["classes"] or None,
    maintenance_class_descriptions=_maint["class_descriptions"],
)
print(f"[webapp] Gemma context: {len(CATEGORIES)} categories, "
      f"{len(MAINTENANCE_TYPES)} maintenance types, "
      f"{len(_maint['descriptions'])} maintenance descriptions from {_MR_PATH.name}")


def run_acf_analyser(diff_data, prior_changes=None):
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
        categories=CATEGORIES,
        maintenance_types=MAINTENANCE_TYPES,
        leaf_to_class=LEAF_TO_CLASS,
        leaf_to_base=LEAF_TO_BASE,
        system_prompt=SYSTEM_PROMPT,
        api_key=api_key,
        rate_limiter=rate_limiter,
        prior_changes=prior_changes,
    )
    return record

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/stream')
async def stream():
    repo_url = request.args.get('repo', '')
    # Force a re-run even when a cached result exists
    refresh = bool(request.args.get('refresh'))

    async def generate_events():
        try:
            slug = repo_slug(repo_url)
            model_dir = MEMORY_DIR / slug / "single" / _safe_name(DEFAULT_MODEL)
            cache_file = model_dir / "results.json"

            # Load any prior progress to resume interrupted runs
            cached = None
            if cache_file.exists() and not refresh:
                try:
                    cached = json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception as read_err:
                    print(f"[memory] could not read cache, restarting: {read_err}")
                    cached = None

            # Complete cache, replays everything and stop
            if cached and cached.get("complete"):
                yield f"event: status\ndata: {json.dumps({'message': 'Repository already analysed — loading results from memory...'})}\n\n"
                cached_diffs = cached.get("diffs")
                if cached_diffs:
                    with open(args.diffs_json, "w", encoding="utf-8") as f:
                        json.dump(cached_diffs, f, indent=2)
                for payload in cached.get("results", []):
                    yield f"event: diff_result\ndata: {json.dumps(payload)}\n\n"
                yield f"event: complete\ndata: {json.dumps({'message': 'Analysis of the entire history completed!'})}\n\n"
                return

            # Determine the work set: RESUME from a partial run, else fetch
            if cached and cached.get("diffs"):
                # Reuse the previously fetched diffs and pick up where we left off
                diffs_list = cached["diffs"]
                collected = cached.get("results", [])
                processed_ids = {p.get("diff_id") for p in collected}
                yield f"event: status\ndata: {json.dumps({'message': f'Resuming analysis — {len(processed_ids)}/{len(diffs_list)} already done...'})}\n\n"
                # Re-emit the already-analysed diffs so the UI is fully populated
                for payload in collected:
                    yield f"event: diff_result\ndata: {json.dumps(payload)}\n\n"
            else:
                yield f"event: status\ndata: {json.dumps({'message': 'Searching for AGENTS.md or CLAUDE.md history on GitHub...'})}\n\n"
                diffs_list = await fetch_github_diffs(repo_url, max_diffs=50)
                if not diffs_list:
                    yield f"event: stream_error\ndata: {json.dumps({'message': 'No textual diffs found.'})}\n\n"
                    return
                collected = []
                processed_ids = set()

            # Save the diffs to disk so multi-model validation can reuse them
            with open(args.diffs_json, "w", encoding="utf-8") as f:
                json.dump(diffs_list, f, indent=2)

            total_diffs = len(diffs_list)
            model_dir.mkdir(parents=True, exist_ok=True)

            _MAX_PRIOR = 3
            prior_changes_by_id: dict[str, list[dict[str, str]]] = {}
            _diffs_by_file: dict[tuple[str, str], list[dict]] = {}
            for _d in diffs_list:
                _key = (str(_d.get("repo", "")), str(_d.get("filename", "unknown")))
                _diffs_by_file.setdefault(_key, []).append(_d)
            for _items in _diffs_by_file.values():
                _ordered = sorted(_items, key=lambda x: str(x.get("timestamp", "")))
                _history: list[dict[str, str]] = []
                for _d in _ordered:
                    _did = str(_d.get("diff_id", ""))
                    prior_changes_by_id[_did] = _history[-_MAX_PRIOR:]
                    _history = _history + [{
                        "commit_date": str(_d.get("timestamp", "")),
                        "commit_message": str(_d.get("commit_message", "")),
                    }]

            def _save_progress(complete: bool) -> None:
                # Atomic write so an interruption mid-write can
                # never leave a half-written cache that would break the resume
                try:
                    tmp = cache_file.with_name(cache_file.name + ".tmp")
                    tmp.write_text(
                        json.dumps(
                            {
                                "repo": repo_url,
                                "model": DEFAULT_MODEL,
                                "diffs": diffs_list,
                                "results": collected,
                                "complete": complete,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    os.replace(tmp, cache_file)
                except Exception as save_err:
                    print(f"[memory] failed to persist progress: {save_err}")

            for idx, diff_data in enumerate(diffs_list):
                # Skip diffs already analysed in a previous run
                if diff_data.get("diff_id") in processed_ids:
                    continue

                msg = f"Analysis {idx + 1}/{total_diffs} in progress (Commit: {diff_data['diff_id']})..."
                yield f"event: status\ndata: {json.dumps({'message': msg})}\n\n"

                record = await asyncio.to_thread(
                    run_acf_analyser,
                    diff_data,
                    prior_changes_by_id.get(str(diff_data.get("diff_id", "")), []),
                )
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

                collected.append(payload)
                processed_ids.add(diff_data.get("diff_id"))
                # Persist after every diff so an interruption resumes from here
                _save_progress(complete=False)
                yield f"event: diff_result\ndata: {json.dumps(payload)}\n\n"

            _save_progress(complete=True)
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

    repo_url = request.args.get('repo', '')
    refresh = bool(request.args.get('refresh'))

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

            # Per-repo caching: stable report folder keyed by the repo slug so a
            # second validation of the same link reuses the stored images/summary
            try:
                slug = repo_slug(repo_url)
            except ValueError:
                slug = None
            report_id = slug or timestamp
            report_dir = STATIC_REPORTS_DIR / report_id
            summary_file = report_dir / "summary.json"

            if slug and summary_file.exists() and not refresh:
                yield f"event: multi_status\ndata: {json.dumps({'message': 'Repository already validated — loading cached multi-model report...'})}\n\n"
                # Re-emit a per-model event so the frontend rebuilds the 3 model
                # cards exactly like a fresh run would
                for model in models_to_run:
                    yield f"event: multi_status\ndata: {json.dumps({'message': f'[{model}] DONE -> OK (cached)'})}\n\n"
                yield f"event: multi_complete\ndata: {json.dumps({'message': 'Loaded cached validation.', 'report_id': report_id})}\n\n"
                return

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
                        "--temperature", str(args.temperature),
                        "--env-file", str(env_path.resolve()),
                        "--ollama-base-url", base_url,
                        # Align the study models to the Gemma single-flow config so the
                        # vs-Gemma comparison isolates model differences, not config
                        "--max-diff-chars", str(args.max_diff_chars),
                        "--max-tokens", str(args.max_tokens),
                    ]
                    if args.use_format_schema:
                        cmd.append("--use-format-schema")
                    if args.no_prefill:
                        cmd.append("--no-prefill")
                    
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

                # Archive each model's raw output under memory/<repo>/multi/<model>/
                # so the per-repo history is kept alongside the cached report
                if slug:
                    for model in models_to_run:
                        src = output_dir / sanitize_model_name(model) / f"primary_{timestamp}.jsonl"
                        if src.exists():
                            dst_dir = MEMORY_DIR / slug / "multi" / _safe_name(model)
                            dst_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                shutil.copy2(src, dst_dir / src.name)
                            except Exception as copy_err:
                                print(f"[memory] copy failed for {model}: {copy_err}")

                # Build a Gemma reference file from the webapp's own cached results
                # for this repo, so the agreement script can benchmark
                # the 3 study models against Gemma on the exact same diffs
                gemma_cache = MEMORY_DIR / (slug or "_") / "single" / _safe_name(DEFAULT_MODEL) / "results.json"
                if slug and gemma_cache.exists():
                    try:
                        payloads = json.loads(gemma_cache.read_text(encoding="utf-8")).get("results", [])
                        ref_dir = output_dir / sanitize_model_name(DEFAULT_MODEL)
                        ref_dir.mkdir(parents=True, exist_ok=True)
                        with (ref_dir / f"primary_{timestamp}.jsonl").open("w", encoding="utf-8") as fh:
                            for p in payloads:
                                cats = p.get("results", [])
                                rec = {
                                    "diff_id": p.get("diff_id", ""),
                                    "primary": {"primary_category": cats[0]["category"] if cats else ""},
                                    "category_scores": {c["category"]: c["score"] for c in cats},
                                    "maintenance": p.get("maintenance", {}),
                                }
                                fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
                        await queue.put(f"event: multi_status\ndata: {json.dumps({'message': f'[Analysis] Prepared Gemma reference ({len(payloads)} diffs) for benchmarking.'})}\n\n")
                    except Exception as ref_err:
                        print(f"[reference] failed to build Gemma reference file: {ref_err}")

                # Agreement Analysis Phase
                await queue.put(f"event: multi_status\ndata: {json.dumps({'message': '[Analysis] Compiling inter-model agreement and ambiguity metrics...'})}\n\n")

                report_dir.mkdir(parents=True, exist_ok=True)

                cmd_agr = [
                    sys.executable, "-u", str(agreement_script.resolve()),
                    "--timestamp", timestamp, "--output-dir", str(output_dir.resolve()),
                    "--report-dir", str(report_dir.resolve()),
                    "--reference", DEFAULT_MODEL,
                    # The webapp renders agreement from summary.json, not PNGs, so
                    # skip chart generation entirely
                    "--no-charts", "--models"
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
                        await queue.put(f"event: multi_complete\ndata: {json.dumps({'message': 'Validation complete!', 'report_id': report_id})}\n\n")
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