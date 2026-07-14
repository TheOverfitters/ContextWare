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

# Load label descriptions and examples from a JSON file, supporting multiple possible shapes for flexibility.
def load_label_descriptions(
    label_file: Path | None,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    if not label_file or not label_file.exists():
        return [], {}, {}
    data = json.loads(label_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Label descriptions must be a JSON object with a 'categories' list")
    raw_categories = data.get("categories")
    if not isinstance(raw_categories, list):
        raise ValueError("Label descriptions must contain a 'categories' list")

    categories: list[str] = []
    descriptions: dict[str, str] = {}
    examples: dict[str, str] = {}
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
        # Support both "examples" (array, preferred) and "example" (string).
        examples_list = item.get("examples")
        example_str = item.get("example")
        if isinstance(examples_list, list):
            parts = [e.strip() for e in examples_list if isinstance(e, str) and e.strip()]
            if parts:
                examples[cleaned_name] = " | ".join(
                    f"Ex.{i + 1}: {p}" for i, p in enumerate(parts)
                )
        elif isinstance(example_str, str) and example_str.strip():
            examples[cleaned_name] = example_str.strip()
    return categories, descriptions, examples


# Load the maintenance-type taxonomy (the "why" of a change) from
# modification_request.json. This axis is ORTHOGONAL to the ACF categories:
# it captures the REASON for the modification per ISO/IEC/IEEE 14764:2022
# (corrective / preventive / adaptive / additive / perfective), with the
# dual-parented "adaptive" split into two explicit leaf targets.
def load_maintenance_descriptions(
    mr_file: Path | None,
) -> dict[str, Any]:
    """Parse modification_request.json into prompt-ready structures.

    Returns a dict with:
      * ``types``            -- ordered list of leaf-type names (the labels).
      * ``descriptions``     -- {leaf_name: description}.
      * ``examples``         -- {leaf_name: "Ex.1: ... | Ex.2: ..."}.
      * ``leaf_to_class``    -- {leaf_name: "Correction"|"Enhancement"}.
      * ``leaf_to_base``     -- {leaf_name: base_type} (collapses the two
                                "adaptive (...)" leaves back to "adaptive").
      * ``classes``          -- ordered list of parent-class names.
      * ``class_descriptions`` -- {class_name: description}.

    Every mapping is empty when the file is missing so callers can fall back
    to the defaults defined in acf_prompt.
    """
    empty: dict[str, Any] = {
        "types": [],
        "descriptions": {},
        "examples": {},
        "leaf_to_class": {},
        "leaf_to_base": {},
        "classes": [],
        "class_descriptions": {},
    }
    if not mr_file or not mr_file.exists():
        return empty
    data = json.loads(mr_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("modification_request.json must be a JSON object")

    def _first_class(value: Any) -> str:
        # ``class`` is a string after the adaptive split, but tolerate a list.
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            return first.strip() if isinstance(first, str) else ""
        return ""

    types: list[str] = []
    descriptions: dict[str, str] = {}
    examples: dict[str, str] = {}
    leaf_to_class: dict[str, str] = {}
    leaf_to_base: dict[str, str] = {}
    for item in data.get("maintenance_types", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        cleaned = name.strip()
        types.append(cleaned)
        desc = item.get("description")
        if isinstance(desc, str) and desc.strip():
            descriptions[cleaned] = desc.strip()
        examples_list = item.get("examples")
        if isinstance(examples_list, list):
            parts = [e.strip() for e in examples_list if isinstance(e, str) and e.strip()]
            if parts:
                examples[cleaned] = " | ".join(
                    f"Ex.{i + 1}: {p}" for i, p in enumerate(parts)
                )
        leaf_to_class[cleaned] = _first_class(item.get("class"))
        base = item.get("base_type")
        leaf_to_base[cleaned] = base.strip() if isinstance(base, str) and base.strip() else cleaned

    classes: list[str] = []
    class_descriptions: dict[str, str] = {}
    for item in data.get("maintenance_classes", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        cleaned = name.strip()
        classes.append(cleaned)
        desc = item.get("description")
        if isinstance(desc, str) and desc.strip():
            class_descriptions[cleaned] = desc.strip()

    return {
        "types": types,
        "descriptions": descriptions,
        "examples": examples,
        "leaf_to_class": leaf_to_class,
        "leaf_to_base": leaf_to_base,
        "classes": classes,
        "class_descriptions": class_descriptions,
    }


def load_categories(categories_csv: str | None, default_categories: list[str] | None = None) -> list[str]:
    fallback_categories = default_categories or DEFAULT_CATEGORIES
    if categories_csv:
        parsed = [item.strip() for item in categories_csv.split(",") if item.strip()]
        return parsed or fallback_categories
    return fallback_categories

# Load diffs from an ACF JSON file
def load_diffs_from_acf_data(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and "acf_commits" in data:
        diffs: list[dict[str, Any]] = []
        for commit in data.get("acf_commits", []):
            commit_hash = str(commit.get("hash", ""))
            # The ACF json stores the message under "commit_message"; keep
            # "message" as a fallback for older/alternative shapes.
            commit_message = str(commit.get("commit_message") or commit.get("message") or "")
            timestamp = str(
                commit.get("timestamp")
                or commit.get("commit_date")
                or commit.get("date")
                or ""
            )
            # Parent (previous) commit: needed to link the file's prior state,
            # e.g. to disambiguate adaptive-correction vs adaptive-enhancement.
            base_sha = str(commit.get("base_sha") or commit.get("parent") or "")
            compare_url = str(commit.get("url") or "")
            for entry in commit.get("acf_files", []):
                patch = entry.get("patch")
                if not patch:
                    continue
                filename = str(entry.get("filename", ""))
                diffs.append(
                    {
                        "diff_id": f"{commit_hash}:{filename}",
                        "commit_hash": commit_hash,
                        "base_sha": base_sha,
                        "compare_url": compare_url,
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

# Repo label extraction from JSONL/JSON rows
def _repo_label_from_jsonl_row(row: dict[str, Any]) -> str:
    """Build a '<owner>/<name>' label from a JSONL row's owner/name fields."""
    owner = row.get("repository_owner") or row.get("owner") or ""
    name = row.get("repository_name") or row.get("name") or ""
    if owner and name:
        return f"{owner}/{name}"
    return str(name or owner or "")

# Diff text extraction from JSONL rows, supporting multiple possible field names and structures for flexibility.
def _diff_text_from_jsonl_row(row: dict[str, Any]) -> str:
    """Extract the patch text from a JSONL row, tolerating a few field names."""
    diff_obj = row.get("diff")
    if isinstance(diff_obj, dict):
        patch = diff_obj.get("patch")
        if isinstance(patch, str) and patch:
            return patch
    # Fallbacks for alternative JSONL shapes.
    for key in ("patch", "diff_text", "patch_text"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    added = row.get("added_lines") or (diff_obj or {}).get("added_lines")
    removed = row.get("removed_lines") or (diff_obj or {}).get("removed_lines")
    if isinstance(added, list) or isinstance(removed, list):
        rendered: list[str] = []
        for line in removed or []:
            rendered.append(f"-{line}")
        for line in added or []:
            rendered.append(f"+{line}")
        if rendered:
            return "\n".join(rendered)
    return ""

# Main diff loading function that supports both ACF JSON and JSONL formats, auto-detecting based on file extension
def load_diffs_from_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file (one diff record per line) into canonical diff dicts.

    Each row is expected to have at minimum a ``diff`` sub-object with
    ``filename``/``patch`` fields, plus a commit identifier (``head_sha``
    or ``content_commit_sha``). Rows that fail to parse are skipped with
    no exception so a single malformed line doesn't abort the run.
    """
    diffs: list[dict[str, Any]] = []
    repo_label = ""
    for line_number, raw_line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if not repo_label:
            repo_label = _repo_label_from_jsonl_row(row)

        diff_obj = row.get("diff") if isinstance(row.get("diff"), dict) else {}
        filename = (
            diff_obj.get("filename")
            or row.get("file_path")
            or row.get("filename")
            or ""
        )
        diff_text = _diff_text_from_jsonl_row(row)
        if not diff_text:
            # Skip rows with no patchable content (e.g. pure deletions / empty).
            continue

        commit_hash = (
            row.get("head_sha")
            or row.get("content_commit_sha")
            or row.get("commit_hash")
            or row.get("hash")
            or ""
        )
        # Build a stable diff_id even if the commit hash is missing.
        diff_id = str(
            row.get("diff_id")
            or f"{commit_hash or 'jsonl'}:{filename}:{line_number}"
        )

        # Commit date lives under "commit_date" in this JSONL shape; keep a few
        # fallbacks for alternative feeds. Empty string only if none are present.
        timestamp = str(
            row.get("commit_date")
            or row.get("timestamp")
            or row.get("committed_date")
            or row.get("date")
            or ""
        )
        diffs.append(
            {
                "diff_id": diff_id,
                "commit_hash": str(commit_hash),
                "commit_message": str(row.get("commit_message", "")),
                "timestamp": timestamp,
                "commit_author": str(row.get("commit_author", "")),
                "filename": str(filename),
                "diff_text": diff_text,
                "repo": _repo_label_from_jsonl_row(row),
                "base_sha": row.get("base_sha", ""),
                "head_sha": row.get("head_sha", ""),
                "compare_url": row.get("compare_html_url", ""),
            }
        )
    # Attach the first-seen repo label so the output path uses it.
    if diffs and not diffs[0].get("repo"):
        diffs[0]["repo"] = repo_label
    return diffs


def load_repo_diffs(json_path: Path) -> tuple[str, list[dict[str, Any]]]:
    if json_path.suffix.lower() == ".jsonl":
        diffs = load_diffs_from_jsonl(json_path)
        repo_label = diffs[0].get("repo", "") if diffs else json_path.stem
        return repo_label, diffs

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