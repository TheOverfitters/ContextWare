from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from acf_prompt import DEFAULT_CATEGORIES


def load_dotenv_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_ollama_api_key(env_path: Path) -> str:
    load_dotenv_file(env_path)
    return os.getenv("OLLAMA_API_KEY", "").strip()


def load_label_descriptions(label_file: Path | None) -> tuple[list[str], dict[str, str]]:
    if not label_file or not label_file.exists():
        return [], {}
    data = json.loads(label_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Label descriptions must be a JSON object with a 'categories' list")
    raw_categories = data.get("categories")
    if not isinstance(raw_categories, list):
        raise ValueError("Label descriptions must contain a 'categories' list")

    categories: list[str] = []
    descriptions: dict[str, str] = {}
    for item in raw_categories:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        cleaned_name = name.strip()
        categories.append(cleaned_name)
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            descriptions[cleaned_name] = description.strip()
    return categories, descriptions


def load_categories(categories_csv: str | None, default_categories: list[str] | None = None) -> list[str]:
    fallback_categories = default_categories or DEFAULT_CATEGORIES
    if categories_csv:
        parsed = [item.strip() for item in categories_csv.split(",") if item.strip()]
        return parsed or fallback_categories
    return fallback_categories


def load_diffs_from_acf_data(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "acf_commits" in data:
        diffs: list[dict[str, Any]] = []
        for commit in data.get("acf_commits", []):
            commit_hash = str(commit.get("hash", ""))
            commit_message = str(commit.get("message", ""))
            timestamp = str(commit.get("timestamp", ""))
            for entry in commit.get("acf_files", []):
                patch = entry.get("patch")
                if not patch:
                    continue
                filename = str(entry.get("filename", ""))
                diffs.append(
                    {
                        "diff_id": f"{commit_hash}:{filename}",
                        "commit_hash": commit_hash,
                        "commit_message": commit_message,
                        "timestamp": timestamp,
                        "filename": filename,
                        "diff_text": str(patch),
                    }
                )
        return diffs

    if isinstance(data, list):
        # List of repo objects (each has "acf_commits")
        if data and isinstance(data[0], dict) and "acf_commits" in data[0]:
            diffs: list[dict[str, Any]] = []
            for repo_obj in data:
                if not isinstance(repo_obj, dict):
                    continue
                repo_name = str(repo_obj.get("repo") or repo_obj.get("repository") or "")
                for diff in load_diffs_from_acf_data(repo_obj):
                    diff["repo"] = repo_name
                    diffs.append(diff)
            return diffs

        # List of flat diff objects
        diffs = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            diff_text = str(item.get("diff_text") or item.get("patch") or "")
            if not diff_text:
                continue
            diff_id = str(item.get("diff_id") or f"diff-{idx}")
            record = dict(item)
            record["diff_id"] = diff_id
            record["diff_text"] = diff_text
            diffs.append(record)
        return diffs

    raise ValueError("Unsupported JSON format for diffs input")


def load_diffs_from_acf_json(json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return load_diffs_from_acf_data(data)


def load_repo_diffs(json_path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    repo_label = ""
    if isinstance(data, dict):
        for key in ("repo", "repository", "repo_name"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                repo_label = value.strip()
                break
    return repo_label, load_diffs_from_acf_data(data)


def truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...TRUNCATED..."


def sanitize_model_name(model: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._-")
    return cleaned or "model"


def sanitize_repo_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-")
    return cleaned or "repo"


def get_run_timestamp(timestamp_override: str | None) -> str:
    if timestamp_override:
        return timestamp_override
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_output_paths(output_dir: Path, model: str, timestamp_str: str) -> tuple[Path, Path]:
    model_dir = output_dir / sanitize_model_name(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    primary_path = model_dir / f"primary_{timestamp_str}.jsonl"
    eval_path = model_dir / f"eval_{timestamp_str}.jsonl"
    return primary_path, eval_path


def build_repo_output_paths(
    output_dir: Path,
    model: str,
    timestamp_str: str,
    repo_label: str,
    repo_index: int,
) -> tuple[Path, Path]:
    base_label = sanitize_repo_label(repo_label) if repo_label else "repo"
    repo_dir = output_dir / f"{repo_index}-{base_label}"
    return build_output_paths(repo_dir, model, timestamp_str)


def load_processed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    processed = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        diff_id = record.get("diff_id")
        if isinstance(diff_id, str) and diff_id:
            processed.add(diff_id)
    return processed