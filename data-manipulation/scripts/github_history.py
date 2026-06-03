from __future__ import annotations

import argparse
import csv
import json
import os
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

WAIT_ON_RATE_LIMIT = False
MAX_RATE_LIMIT_WAIT_S = 900


def load_env_files(paths: Iterable[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


def _rate_limit_wait_seconds(status: int, data: object, headers: dict) -> Optional[int]:
    if status not in (403, 429):
        return None
    message = ""
    if isinstance(data, dict):
        message = str(data.get("message") or "")
    lowered = message.lower()
    if "rate limit" not in lowered and status != 429:
        return None

    retry_after_raw = headers.get("Retry-After")
    if retry_after_raw:
        try:
            retry_after = int(float(retry_after_raw))
            return max(1, retry_after)
        except (TypeError, ValueError):
            pass

    reset_raw = headers.get("X-RateLimit-Reset")
    if reset_raw:
        try:
            reset_epoch = int(float(reset_raw))
            return max(1, reset_epoch - int(time.time()) + 1)
        except (TypeError, ValueError):
            pass

    return 60


def rest_get_json(url: str, token: Optional[str]) -> Tuple[dict, int, dict]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "contextware-history-script",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    waited_total = 0
    max_retries = 5
    retry_attempt = 0
    while True:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body) if body else {}
                return data, resp.status, dict(resp.headers)
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError):
            retry_attempt += 1
            if retry_attempt >= max_retries:
                return {"message": "network error after retries"}, 0, {}
            wait = min(2 ** retry_attempt, 30)
            print(f"Network error on {url}, retrying in {wait}s ({max_retries - retry_attempt} left)...")
            time.sleep(wait)
            continue
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {"message": body}
            response_headers = dict(exc.headers)

            wait_s = _rate_limit_wait_seconds(exc.code, data, response_headers)
            if not WAIT_ON_RATE_LIMIT or wait_s is None:
                return data, exc.code, response_headers
            if MAX_RATE_LIMIT_WAIT_S > 0 and waited_total + wait_s > MAX_RATE_LIMIT_WAIT_S:
                return data, exc.code, response_headers

            print(f"GitHub API rate limited. Waiting {wait_s}s before retry...")
            time.sleep(wait_s)
            waited_total += wait_s


def to_ascii(value: str) -> str:
    return value.encode("ascii", "replace").decode("ascii")


def get_default_branch(owner: str, repo: str, token: Optional[str]) -> Optional[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data, status, _headers = rest_get_json(url, token)
    if status != 200 or not isinstance(data, dict):
        return None
    return data.get("default_branch")


def get_repo_status(
    owner: str, repo: str, token: Optional[str]
) -> Tuple[int, Optional[str]]:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data, status, _headers = rest_get_json(url, token)
    message = None
    if isinstance(data, dict):
        message = data.get("message")
    return status, message


def check_owner_exists(owner: str) -> Optional[bool]:
    url = f"https://api.github.com/users/{owner}"
    _data, status, _headers = rest_get_json(url, None)
    if status == 200:
        return True
    if status == 404:
        return False
    return None


def fetch_commits_page(
    owner: str,
    repo: str,
    file_path: str,
    ref: Optional[str],
    page: int,
    token: Optional[str],
) -> Tuple[object, int, dict]:
    params = {
        "path": file_path,
        "per_page": "100",
        "page": str(page),
    }
    if ref:
        params["sha"] = ref
    url = (
        f"https://api.github.com/repos/{owner}/{repo}/commits?"
        + urllib.parse.urlencode(params, safe="/")
    )
    return rest_get_json(url, token)


def fetch_repo_commits_page(
    owner: str, repo: str, ref: Optional[str], page: int, token: Optional[str]
) -> Tuple[object, int, dict]:
    params = {
        "per_page": "100",
        "page": str(page),
    }
    if ref:
        params["sha"] = ref
    url = (
        f"https://api.github.com/repos/{owner}/{repo}/commits?"
        + urllib.parse.urlencode(params, safe="/")
    )
    return rest_get_json(url, token)


def collect_commit_window(
    owner: str,
    repo: str,
    file_path: str,
    ref: Optional[str],
    target_sha: str,
    needed: int,
    token: Optional[str],
    sleep_s: float,
    debug: bool,
) -> Tuple[List[str], Optional[dict]]:
    commits: List[str] = []
    found_index: Optional[int] = None
    page = 1
    last_headers: dict = {}

    while True:
        data, status, headers = fetch_commits_page(
            owner=owner,
            repo=repo,
            file_path=file_path,
            ref=ref,
            page=page,
            token=token,
        )
        last_headers = headers

        if status != 200:
            message = None
            if isinstance(data, dict):
                message = data.get("message")
            if debug:
                print(
                    f"commit list failed: {owner}/{repo} {file_path} ref={ref} "
                    f"status={status} message={message}"
                )
            return [], {"status": status, "message": message}

        if not isinstance(data, list) or not data:
            break

        for item in data:
            sha = item.get("sha")
            if not sha:
                continue
            commits.append(sha)
            if sha == target_sha and found_index is None:
                found_index = len(commits) - 1

        if found_index is not None and len(commits) >= found_index + needed:
            break

        page += 1
        if sleep_s:
            time.sleep(sleep_s)

        remaining = last_headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            break

    if found_index is None:
        return [], {"status": "commit_not_found", "message": "not_in_history"}

    window = commits[found_index : found_index + needed]
    return window, None


def collect_window_with_fallback(
    owner: str,
    repo: str,
    file_path: str,
    ref: Optional[str],
    target_sha: str,
    needed: int,
    token: Optional[str],
    sleep_s: float,
    debug: bool,
) -> Tuple[List[str], Optional[dict], Optional[str]]:
    commits, commit_error = collect_commit_window(
        owner=owner,
        repo=repo,
        file_path=file_path,
        ref=ref,
        target_sha=target_sha,
        needed=needed,
        token=token,
        sleep_s=sleep_s,
        debug=debug,
    )

    if commit_error and commit_error.get("status") == 404 and ref:
        fallback_ref = get_default_branch(owner, repo, token)
        if fallback_ref:
            if debug:
                print(
                    f"Retrying with default branch ref={fallback_ref} "
                    f"for {owner}/{repo} {file_path}"
                )
            commits, commit_error = collect_commit_window(
                owner=owner,
                repo=repo,
                file_path=file_path,
                ref=fallback_ref,
                target_sha=target_sha,
                needed=needed,
                token=token,
                sleep_s=sleep_s,
                debug=debug,
            )
            if not commit_error:
                ref = fallback_ref

    return commits, commit_error, ref


def collect_latest_commits(
    owner: str,
    repo: str,
    file_path: str,
    ref: Optional[str],
    needed: int,
    token: Optional[str],
    sleep_s: float,
    debug: bool,
) -> Tuple[List[str], Optional[dict]]:
    commits: List[str] = []
    page = 1
    last_headers: dict = {}

    while len(commits) < needed:
        data, status, headers = fetch_commits_page(
            owner=owner,
            repo=repo,
            file_path=file_path,
            ref=ref,
            page=page,
            token=token,
        )
        last_headers = headers

        if status != 200:
            message = None
            if isinstance(data, dict):
                message = data.get("message")
            if debug:
                print(
                    f"commit list failed: {owner}/{repo} {file_path} ref={ref} "
                    f"status={status} message={message}"
                )
            return [], {"status": status, "message": message}

        if not isinstance(data, list) or not data:
            break

        for item in data:
            sha = item.get("sha")
            if sha:
                commits.append(sha)
                if len(commits) >= needed:
                    break

        page += 1
        if sleep_s:
            time.sleep(sleep_s)

        remaining = last_headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            break

    return commits, None


def collect_latest_with_fallback(
    owner: str,
    repo: str,
    file_path: str,
    ref: Optional[str],
    needed: int,
    token: Optional[str],
    sleep_s: float,
    debug: bool,
) -> Tuple[List[str], Optional[dict], Optional[str]]:
    commits, commit_error = collect_latest_commits(
        owner=owner,
        repo=repo,
        file_path=file_path,
        ref=ref,
        needed=needed,
        token=token,
        sleep_s=sleep_s,
        debug=debug,
    )

    if commit_error and commit_error.get("status") == 404 and ref:
        fallback_ref = get_default_branch(owner, repo, token)
        if fallback_ref:
            if debug:
                print(
                    f"Retrying with default branch ref={fallback_ref} "
                    f"for {owner}/{repo} {file_path}"
                )
            commits, commit_error = collect_latest_commits(
                owner=owner,
                repo=repo,
                file_path=file_path,
                ref=fallback_ref,
                needed=needed,
                token=token,
                sleep_s=sleep_s,
                debug=debug,
            )
            if not commit_error:
                ref = fallback_ref

    return commits, commit_error, ref


def collect_repo_commits(
    owner: str,
    repo: str,
    ref: Optional[str],
    token: Optional[str],
    sleep_s: float,
    debug: bool,
    max_commits: int,
) -> Tuple[List[dict], Optional[dict]]:
    commits: List[dict] = []
    page = 1
    last_headers: dict = {}

    while True:
        data, status, headers = fetch_repo_commits_page(
            owner=owner,
            repo=repo,
            ref=ref,
            page=page,
            token=token,
        )
        last_headers = headers

        if status != 200:
            message = None
            if isinstance(data, dict):
                message = data.get("message")
            if debug:
                print(
                    f"repo commit list failed: {owner}/{repo} ref={ref} "
                    f"status={status} message={message}"
                )
            return [], {"status": status, "message": message}

        if not isinstance(data, list) or not data:
            break

        for item in data:
            commit = item.get("commit") or {}
            author = commit.get("author") or {}
            committer = commit.get("committer") or {}
            date = author.get("date") or committer.get("date") or ""
            name = author.get("name") or committer.get("name") or ""
            message = (commit.get("message") or "").splitlines()[0]
            commits.append(
                {
                    "sha": item.get("sha") or "",
                    "date": date,
                    "author": name,
                    "message": message,
                    "html_url": item.get("html_url") or "",
                }
            )
            if max_commits and len(commits) >= max_commits:
                return commits, None

        page += 1
        if sleep_s:
            time.sleep(sleep_s)

        remaining = last_headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            break

    return commits, None


def collect_repo_commits_with_fallback(
    owner: str,
    repo: str,
    ref: Optional[str],
    token: Optional[str],
    sleep_s: float,
    debug: bool,
    max_commits: int,
) -> Tuple[List[dict], Optional[dict], Optional[str], str]:
    auth_mode = "token" if token else "anonymous"
    commits, commit_error = collect_repo_commits(
        owner=owner,
        repo=repo,
        ref=ref,
        token=token,
        sleep_s=sleep_s,
        debug=debug,
        max_commits=max_commits,
    )

    if commit_error and commit_error.get("status") == 404 and ref:
        fallback_ref = get_default_branch(owner, repo, token)
        if fallback_ref:
            if debug:
                print(
                    f"Retrying repo history with default branch ref={fallback_ref} "
                    f"for {owner}/{repo}"
                )
            commits, commit_error = collect_repo_commits(
                owner=owner,
                repo=repo,
                ref=fallback_ref,
                token=token,
                sleep_s=sleep_s,
                debug=debug,
                max_commits=max_commits,
            )
            if not commit_error:
                ref = fallback_ref

    message = (commit_error or {}).get("message") or ""
    rate_limited = (
        (commit_error or {}).get("status") == 403
        and "rate limit" in message.lower()
    )

    if commit_error and token and not rate_limited:
        public_status, _public_message = get_repo_status(owner, repo, None)
        if public_status == 200:
            if debug:
                print(f"Retrying repo history without token: {owner}/{repo}")
            commits, commit_error = collect_repo_commits(
                owner=owner,
                repo=repo,
                ref=ref,
                token=None,
                sleep_s=sleep_s,
                debug=debug,
                max_commits=max_commits,
            )
            if not commit_error:
                auth_mode = "public_no_token"

    return commits, commit_error, ref, auth_mode


def parse_patch_lines(patch: Optional[str]) -> Tuple[List[str], List[str]]:
    """Parse a unified diff patch into added and removed instruction lines.

    Skips hunk headers (@@) and file headers (+++/---). Returns stripped text
    so callers get clean sentences ready for LLM processing.
    """
    if not patch:
        return [], []
    added: List[str] = []
    removed: List[str] = []
    for raw_line in patch.splitlines():
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            continue
        if raw_line.startswith("+"):
            text = raw_line[1:].strip()
            if text:
                added.append(text)
        elif raw_line.startswith("-"):
            text = raw_line[1:].strip()
            if text:
                removed.append(text)
    return added, removed


def extract_file_diff(compare_json: dict, file_path: str) -> dict:
    files = compare_json.get("files") or []
    for entry in files:
        filename = entry.get("filename")
        prev_name = entry.get("previous_filename")
        if filename == file_path or prev_name == file_path:
            patch = entry.get("patch")
            status = entry.get("status")
            added_lines, removed_lines = parse_patch_lines(patch)
            result = {
                "filename": filename,
                "previous_filename": prev_name,
                "status": status,
                "additions": entry.get("additions"),
                "deletions": entry.get("deletions"),
                "changes": entry.get("changes"),
                "patch": patch,
                "added_lines": added_lines,
                "removed_lines": removed_lines,
            }
            if patch is None:
                if status == "renamed":
                    result["patch_missing_reason"] = "file_renamed_no_content_change"
                elif status == "removed":
                    result["patch_missing_reason"] = "file_deleted"
                elif status == "added":
                    result["patch_missing_reason"] = "file_added_too_large"
                else:
                    result["patch_missing_reason"] = "file_too_large_or_binary"
            return result
    return {"status": "not_found"}


def compare_commits(
    owner: str,
    repo: str,
    base_sha: str,
    head_sha: str,
    token: Optional[str],
) -> Tuple[dict, int, dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/compare/{base_sha}...{head_sha}"
    return rest_get_json(url, token)


def iter_rows(csv_path: str, max_rows: int) -> Iterable[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            if max_rows and idx > max_rows:
                break
            yield row


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch file history diffs from GitHub.")
    parser.add_argument(
        "--csv",
        default=os.path.join("dataset", "dataset_cleaned.csv"),
        help="Path to dataset_cleaned.csv",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("outputs", "git_history_diffs.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--windows",
        default="3,5,10",
        help="Comma-separated window sizes",
    )
    parser.add_argument(
        "--target-filename",
        nargs="*",
        default=["CLAUDE.md", "AGENTS.md"],
        help="Filenames to process (space-separated). Default: CLAUDE.md AGENTS.md",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limit number of rows (0 = all)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep between API calls",
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Env var that stores the GitHub token",
    )
    parser.add_argument(
        "--wait-on-rate-limit",
        action="store_true",
        help="Wait and retry when GitHub API rate limits are hit",
    )
    parser.add_argument(
        "--max-rate-limit-wait",
        type=int,
        default=900,
        help="Max seconds to wait per API request for rate-limit reset (0 = unlimited)",
    )
    parser.add_argument(
        "--repo-history-output",
        default="",
        help="Folder where repository history txt files are saved",
    )
    parser.add_argument(
        "--repo-history-max",
        type=int,
        default=0,
        help="Max commits per repository history (0 = all)",
    )
    parser.add_argument(
        "--all-diffs",
        action="store_true",
        help="Fetch all available diffs for each file (no window limit)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    global WAIT_ON_RATE_LIMIT, MAX_RATE_LIMIT_WAIT_S
    WAIT_ON_RATE_LIMIT = args.wait_on_rate_limit
    MAX_RATE_LIMIT_WAIT_S = max(0, args.max_rate_limit_wait)

    load_env_files([".env", os.path.join(".contextWare", ".env")])
    token = os.environ.get(args.token_env)

    windows = [int(x) for x in args.windows.split(",") if x.strip().isdigit()]
    if not args.all_diffs and not windows:
        print("No valid windows provided.")
        return 2
    if args.all_diffs:
        max_steps = 999_999
    else:
        max_steps = max(windows)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_count = 0
    error_count = 0
    repo_history_done: set[str] = set()
    for row in iter_rows(args.csv, args.max_rows):
        if args.target_filename:
            filename = (row.get("filename") or "").strip()
            if filename not in args.target_filename:
                continue

        owner = (row.get("repository_owner") or "").strip()
        repo = (row.get("repository_name") or "").strip()
        file_path = (row.get("file_path") or "").strip()
        content_sha = (row.get("content_commit_sha") or "").strip()
        ref = (row.get("branch") or "").strip() or None

        if not owner or not repo or not file_path or not content_sha:
            if args.debug:
                print("Skipping row with missing fields.")
            continue

        if args.repo_history_output:
            repo_key = f"{owner}/{repo}"
            if repo_key not in repo_history_done:
                repo_history_done.add(repo_key)
                os.makedirs(args.repo_history_output, exist_ok=True)
                repo_ref = ref
                commits, history_error, repo_ref, history_auth = (
                    collect_repo_commits_with_fallback(
                        owner=owner,
                        repo=repo,
                        ref=repo_ref,
                        token=token,
                        sleep_s=args.sleep,
                        debug=args.debug,
                        max_commits=args.repo_history_max,
                    )
                )
                if history_error:
                    if args.debug:
                        print(
                            "Repo history failed: "
                            f"{owner}/{repo} status={history_error.get('status')}"
                        )
                else:
                    safe_repo = f"{owner}__{repo}".replace("/", "__")
                    txt_path = os.path.join(
                        args.repo_history_output, f"{safe_repo}.txt"
                    )
                    with open(txt_path, "w", encoding="utf-8") as handle:
                        header = (
                            f"Repo: {owner}/{repo}\n"
                            f"Branch: {repo_ref or 'default'}\n"
                            f"Auth: {history_auth}\n"
                            f"Total: {len(commits)}\n"
                        )
                        handle.write(header)
                        for item in commits:
                            sha = item.get("sha") or ""
                            date = item.get("date") or ""
                            author = item.get("author") or ""
                            message = item.get("message") or ""
                            url = item.get("html_url") or ""
                            line = f"{sha} | {date} | {author} | {message} | {url}\n"
                            handle.write(line)

        commit_source = "content_commit_sha"
        commits, commit_error, ref = collect_window_with_fallback(
            owner=owner,
            repo=repo,
            file_path=file_path,
            ref=ref,
            target_sha=content_sha,
            needed=max_steps + 1,
            token=token,
            sleep_s=args.sleep,
            debug=args.debug,
        )

        auth_mode = "token" if token else "anonymous"
        message = (commit_error or {}).get("message") or ""
        rate_limited = (
            (commit_error or {}).get("status") == 403
            and "rate limit" in message.lower()
        )

        if commit_error and token and not rate_limited:
            public_status, _public_message = get_repo_status(owner, repo, None)
            if public_status == 200:
                if args.debug:
                    print(
                        "Retrying without token for public repo: "
                        f"{owner}/{repo} {file_path}"
                    )
                commits, commit_error, ref = collect_window_with_fallback(
                    owner=owner,
                    repo=repo,
                    file_path=file_path,
                    ref=ref,
                    target_sha=content_sha,
                    needed=max_steps + 1,
                    token=None,
                    sleep_s=args.sleep,
                    debug=args.debug,
                )
                if not commit_error:
                    auth_mode = "public_no_token"

        if commit_error and commit_error.get("status") == "commit_not_found":
            fallback_token = None if auth_mode == "public_no_token" else token
            commits, latest_error, ref = collect_latest_with_fallback(
                owner=owner,
                repo=repo,
                file_path=file_path,
                ref=ref,
                needed=max_steps + 1,
                token=fallback_token,
                sleep_s=args.sleep,
                debug=args.debug,
            )
            if not latest_error:
                commit_error = None
                commit_source = "latest_file_commit"

        if commit_error:
            access_check = None
            if commit_error.get("status") == 404:
                public_status, _public_message = get_repo_status(owner, repo, None)
                if public_status == 200:
                    access_check = {
                        "repo_access": "public_repo",
                        "note": "commit_list_404_ref_or_path",
                    }
                else:
                    owner_exists = check_owner_exists(owner)
                    if owner_exists is False:
                        access_check = {"repo_access": "owner_not_found"}
                    elif token:
                        access_check = {"repo_access": "access_denied_or_private"}
                    else:
                        access_check = {"repo_access": "repo_missing_or_private"}

                if args.debug and access_check:
                    print(
                        "Repo access check: "
                        f"{owner}/{repo} -> {access_check.get('repo_access')}"
                    )

            record = {
                "repository_owner": owner,
                "repository_name": repo,
                "file_path": file_path,
                "content_commit_sha": content_sha,
                "branch": ref,
                "error": "commit_history_failed",
                "details": commit_error,
                "repo_access": access_check,
                "auth_mode": auth_mode,
                "commit_source": commit_source,
            }
            with open(args.output, "a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")
            error_count += 1

            status = commit_error.get("status")
            if rate_limited:
                if WAIT_ON_RATE_LIMIT:
                    print(
                        "GitHub API rate limit exceeded and max wait reached. "
                        "Stopping early."
                    )
                else:
                    print(
                        "GitHub API rate limit exceeded. Stopping early. "
                        "Use --wait-on-rate-limit to continue after reset."
                    )
                break
            continue

        if len(commits) < 2:
            record = {
                "repository_owner": owner,
                "repository_name": repo,
                "file_path": file_path,
                "content_commit_sha": content_sha,
                "commit_source": commit_source,
                "error": "not_enough_commits",
                "found": len(commits),
            }
            with open(args.output, "a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")
            error_count += 1
            continue

        max_available = min(max_steps, len(commits) - 1)
        for i in range(max_available):
            base_sha = commits[i + 1]
            head_sha = commits[i]
            compare_json, status, _headers = compare_commits(
                owner=owner,
                repo=repo,
                base_sha=base_sha,
                head_sha=head_sha,
                token=None if auth_mode == "public_no_token" else token,
            )

            if status != 200:
                diff_data = {"status": f"compare_failed_{status}"}
            else:
                diff_data = extract_file_diff(compare_json, file_path)

            commit_message = ""
            commit_author = ""
            commit_date = ""
            if status == 200:
                compare_commits_list = compare_json.get("commits") or []
                if compare_commits_list:
                    last_commit = compare_commits_list[-1]
                    commit_obj = last_commit.get("commit") or {}
                    author_obj = commit_obj.get("author") or {}
                    committer_obj = commit_obj.get("committer") or {}
                    commit_message = (commit_obj.get("message") or "").splitlines()[0]
                    commit_author = author_obj.get("name") or committer_obj.get("name") or ""
                    commit_date = author_obj.get("date") or committer_obj.get("date") or ""

            record = {
                "repository_owner": owner,
                "repository_name": repo,
                "file_path": file_path,
                "content_commit_sha": content_sha,
                "commit_source": commit_source,
                "auth_mode": auth_mode,
                "diff_index": i + 1,
                "included_in_windows": [w for w in windows if i + 1 <= w],
                "base_sha": base_sha,
                "head_sha": head_sha,
                "compare_html_url": compare_json.get("html_url"),
                "commit_message": commit_message,
                "commit_author": commit_author,
                "commit_date": commit_date,
                "diff": diff_data,
            }

            with open(args.output, "a", encoding="utf-8") as out:
                out.write(json.dumps(record) + "\n")
            output_count += 1

            if args.sleep:
                time.sleep(args.sleep)

    print(
        f"Wrote {output_count} diff records and {error_count} error records to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
