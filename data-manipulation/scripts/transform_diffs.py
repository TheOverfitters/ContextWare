import json
from collections import defaultdict

INPUT_FILE = "outputs/git_history_diffs.json"
OUTPUT_FILE = "outputs/git_history_diffs.json"

# Read JSONL
entries = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            entries.append(json.loads(line))

# Group by repo, then by (base_sha, head_sha) as a "commit"
repos = defaultdict(lambda: defaultdict(list))
repo_meta = {}

for entry in entries:
    repo_key = f"{entry['repository_owner']}/{entry['repository_name']}"
    commit_key = (entry["base_sha"], entry["head_sha"])

    if repo_key not in repo_meta:
        repo_meta[repo_key] = {
            "acf_patterns": set(),
        }
    escaped = entry['file_path'].replace('.', r'\.')
    repo_meta[repo_key]["acf_patterns"].add(f"^{escaped}$")

    repos[repo_key][commit_key].append(entry)

# Build output structure
output = []

for repo_key in sorted(repos.keys()):
    commit_groups = repos[repo_key]
    meta = repo_meta[repo_key]

    acf_commits = []
    for (base_sha, head_sha), diff_entries in commit_groups.items():
        # Sort by diff_index within each commit group
        diff_entries.sort(key=lambda e: e.get("diff_index", 0))

        first = diff_entries[0]

        acf_files = []
        for entry in diff_entries:
            d = entry["diff"]
            file_info = {
                "filename": d["filename"],
                "status": d["status"],
                "additions": d["additions"],
                "deletions": d["deletions"],
                "changes": d["changes"],
                "patch": d.get("patch"),
            }
            if d.get("previous_filename"):
                file_info["previous_filename"] = d["previous_filename"]
            if d.get("added_lines"):
                file_info["added_lines"] = d["added_lines"]
            if d.get("removed_lines"):
                file_info["removed_lines"] = d["removed_lines"]

            blob_url = f"https://github.com/{repo_key}/blob/{head_sha}/{d['filename']}"
            raw_url = f"https://github.com/{repo_key}/raw/{head_sha}/{d['filename']}"
            file_info["blob_url"] = blob_url
            file_info["raw_url"] = raw_url

            acf_files.append(file_info)

        commit_entry = {
            "hash": head_sha,
            "base_sha": base_sha,
            "diff_index": first.get("diff_index"),
            "included_in_windows": first.get("included_in_windows", []),
            "content_commit_sha": first.get("content_commit_sha"),
            "commit_source": first.get("commit_source"),
            "url": first.get("compare_html_url", f"https://github.com/{repo_key}/commit/{head_sha}"),
            "acf_files": acf_files,
        }
        acf_commits.append(commit_entry)

    # Sort commits by diff_index
    acf_commits.sort(key=lambda c: c.get("diff_index") or 0)

    repo_entry = {
        "repo": repo_key,
        "total_diffs_found": len(acf_commits),
        "acf_patterns": sorted(meta["acf_patterns"]),
        "acf_commits": acf_commits,
    }
    output.append(repo_entry)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Done. {len(output)} repos written to {OUTPUT_FILE}")
for r in output:
    print(f"  {r['repo']}: {r['total_diffs_found']} diffs")
