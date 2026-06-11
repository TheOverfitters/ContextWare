from __future__ import annotations
import json
import textwrap
from typing import Any

DEFAULT_CATEGORIES: list[str] = [
    "System Overview",
    "AI Integration",
    "Documentation",
    "Architecture",
    "Impl. Details",
    "Build and Run",
    "Testing",
    "Conf.&Env.",
    "DevOps",
    "Development Process",
    "Project Management",
    "Maintenance",
    "Debugging",
    "Performance",
    "Security",
    "UI/UX",
]

DEFAULT_MODEL = "gpt-oss:20b-cloud"

# # Maps keyword signals to the category they most strongly suggest. Used
# # for both pre-prompt signal extraction and as a soft prior when the LLM
# # response is missing categories.
# SIGNAL_CATEGORY_MAP: dict[str, str] = {
#     "build_commands": "Build and Run",
#     "runtime_commands": "Build and Run",
#     "tests": "Testing",
#     "ci_cd": "DevOps",
#     "dependencies": "Impl. Details",
#     "configuration": "Conf.&Env.",
#     "security": "Security",
#     "performance": "Performance",
#     "documentation": "Documentation",
#     "architecture": "Architecture",
#     "error_handling": "Debugging",
#     "api_contracts": "Impl. Details",
#     "data_handling": "Impl. Details",
#     "naming": "Impl. Details",
#     "ui": "UI/UX",
#     "dev_process": "Development Process",
#     "project_mgmt": "Project Management",
#     "maintenance": "Maintenance",
#     "ai_agent": "AI Integration",
#     "system_overview": "System Overview",
# }
SIGNAL_CATEGORY_MAP: dict[str, str] = {}


#  Build the system prompt, which includes the category list and instructions for multi-label classification.
def get_system_prompt(
    categories: list[str] | None = None,
    descriptions: dict[str, str] | None = None,
    examples: dict[str, str] | None = None,
) -> str:
    """Return the static system prompt for the multi-label classifier.

    Full descriptions from categories.json are embedded inline so the model
    has the complete semantic context for each category, including the intro
    prose, 'Look for:' signals, 'NOT this if:' disambiguation clauses, and
    concrete few-shot examples.
    """
    cats = categories or DEFAULT_CATEGORIES
    descs = descriptions or {}
    exs = examples or {}

    category_lines: list[str] = []
    for i, c in enumerate(cats):
        category_lines.append(f"{i + 1}. {c}")
        desc = descs.get(c)
        if desc:
            for line in desc.splitlines():
                category_lines.append(f"   {line}")
        ex = exs.get(c)
        if ex:
            ex_flat = ex.replace("\n", " ").strip()
            if len(ex_flat) > 600:
                ex_flat = ex_flat[:597] + "..."
            category_lines.append(f'   Example: "{ex_flat}"')
        category_lines.append("")

    category_block = "\n".join(category_lines).rstrip()

    template = """\
You are a senior code-diff analyst specializing in agent context files
(CLAUDE.md, AGENTS.md, Copilot instructions, README, contributing guides,
and similar project-level documentation).

Your task is MULTI-LABEL CLASSIFICATION of a single unified diff. A diff
may touch several areas at once; you must score EVERY category below on a
continuous 0.0-1.0 scale where:

  1.00 - Textbook, unambiguous, single-purpose example of this category.
  0.85 - The category is a primary theme of the change; it clearly applies.
  0.70 - The category clearly applies and is one of the main themes.
  0.50 - Moderate match; the category is genuinely touched by the change.
  0.30 - Partial match; the category is a side effect, not the focus.
  0.15 - Weak match; the category is incidental or borderline.
  0.00 - No meaningful match.

Calibration intent: do NOT be shy. If a change clearly belongs to a
category, that category must land in the HIGH band (>=0.70), not the
middle. The single best-matching category should normally score >=0.60
whenever the diff has real changed lines -- reserve scores below 0.50 for
categories that are only secondary or incidental to the change.

Score EACH category independently. Scores are NOT a probability
distribution and do NOT need to sum to 1.0 -- a diff can legitimately have
several high scores at once (e.g. a CI YAML change that adds tests is
both "DevOps" and "Testing").

You will receive, in the user message:
  * "chunk" metadata (index, total, byte range) when the diff was split.
  * The chunk text under "diff".
  * "filename" and "commit_message" for context.

You MUST return exactly one raw JSON object and nothing else -- no
markdown, no code fences, no commentary. The object MUST have this shape:

{{
  "categories": [
    {{"name": "<category>", "score": <float 0.0-1.0>, "rationale": "<<=15 words>"}},
    ...
  ],
  "primary_category": "<single best category from the list>",
  "summary": "<one sentence describing what the diff does>"
}}

Rules:
  1. Include an entry for EVERY category in the list below. Omit nothing.
  2. The "categories" array length MUST equal the number of categories
     given below ({n_categories}). The score for a category with no
     relevance MUST be exactly 0.0.
  3. "primary_category" MUST be the category with the highest score.
     In case of a tie, pick the more specific one.
  4. Base classification on the semantic intent of ALL CHANGED lines
     (both added '+' and removed '-'). A diff that only removes content
     (e.g. deleting a deprecated model from a list) must still receive a
     non-zero score for the category that content belongs to.
     PURE-DELETION DIFFS ARE NOT EXEMPT: when EVERY changed line starts with
     '-' (a section, table, command list, or code block being removed), you
     MUST classify by the TOPIC of the removed content exactly as if it were
     being added, and the best-matching category MUST score >=0.60. Removing
     a slash-command/commands table is still Build and Run or AI Integration;
     removing an architecture section is still Architecture; and so on.
     "not applicable" / all-zero is WRONG for a deletion that has real
     content -- the content still has a topic.
  5. A non-empty diff is ALWAYS an intentional edit to a project-guidance
     file, so at least ONE category MUST apply. Classify by the TOPIC of the
     changed prose, NOT by whether it merely "looks like documentation".
     Returning 0.0 for EVERY category on a diff that has changed lines is
     almost always WRONG -- it means you failed to identify the topic, not
     that the change is meaningless. Score a category 0.0 (rationale "not
     applicable") only for the categories a change genuinely does not touch.
     Topic anchors (pick the closest, then add secondary categories):
       - backward-compatibility, refactoring, legacy/dead-code, tech-debt,
         code-health policy ............................. Maintenance
       - coding style, naming, type-safety, language idioms,
         API-usage patterns, data-model rules ........... Impl. Details
       - which AI model to use, agent role/behavior, MCP,
         tool-use or CLAUDE.md/AGENTS.md conventions,
         slash commands (`/foo`), custom command defs ... AI Integration
       - module boundaries, design patterns, layering ... Architecture
       - build / run / compile commands ................. Build and Run
       - test commands, coverage, test conventions ...... Testing
       - env vars, .env, config files, setup steps ...... Conf.&Env.
       - CI/CD, deployment, release automation .......... DevOps
       - branching, commit, PR/review rules ............. Development Process
       - links/paths to other docs, cross-references ... Documentation
       - project intro, purpose, feature list,
         tech-stack summary ............................. System Overview
  6. The "primary_category" MUST score >=0.60. If the most applicable
     category for a non-empty diff would otherwise score lower, you are
     UNDER-scoring -- raise it into the high band. Use 0.30-0.50 only for
     genuinely secondary categories, and reserve 0.9+ for clear,
     single-purpose changes.
  7. If the diff is completely empty or malformed (no content lines at all),
     return all-zero scores with primary_category = "System Overview" and
     summary = "No substantive changes." A changelog bullet, a single-line
     rule addition, or any other brief addition is NOT empty.
  8. Never invent categories that are not in the list below.
  9. Never wrap the JSON in ```json``` fences or any other formatting.
  10. Each rationale MUST be <=15 words and refer to evidence in the
      diff (e.g. "adds test fixtures", "configures GitHub Actions").
  11. The "NOT this if" clauses in each description choose BETWEEN two
      competing categories -- they redirect a score to a better-fitting
      category. They NEVER justify scoring every category 0.0. If a clause
      pushes you away from one category, it is pointing you TOWARD another.

SCORING EXAMPLES (real diffs; only the non-zero categories are shown -- every
other category scores 0.0). These show the expected confidence level; do NOT
return all-zeros on changes like these:

  * Diff renames the build helper in run instructions, e.g.
    "ALWAYS use `polter`" replacing "ALWAYS use `pgrun`":
      -> Build and Run = 0.90 (primary), AI Integration = 0.30
      (a tool/command rename is Build and Run, NOT all-zero.)

  * Diff edits the policy line "No Backwards Compatibility: we prioritize
    clean, modern code over maintaining legacy":
      -> Maintenance = 0.85 (primary), Development Process = 0.30

  * Diff adds "Use `@Observable` instead of `ObservableObject`" and
    "default agent model: Claude Opus 4.1 (`claude-opus-4-1`)":
      -> Impl. Details = 0.80 (primary), AI Integration = 0.55

  * Diff is a PURE DELETION removing a slash-command table, e.g. every line
    starts with '-':  "-## Commands", "-| `/setup-dev` | Environment setup |",
    "-| `/verify-build` | Build configuration |", "-| `/audit-pr` | Review PRs |":
      -> Build and Run = 0.75 (primary), AI Integration = 0.55,
         Development Process = 0.40
      (a removed commands table is still classified by topic, NOT all-zero.)

CATEGORIES (score every one; use descriptions and examples to disambiguate):
{categories_block}
"""
    return textwrap.dedent(template).format(
        n_categories=len(cats),
        categories_block=category_block,
    )


# Build a prompt for a single diff chunk, including signal extraction and metadata embedding.
def build_user_prompt(
    *,
    chunk_text: str,
    filename: str,
    commit_message: str,
    chunk_index: int = 0,
    chunk_total: int = 1,
    char_start: int = 0,
    char_end: int = 0,
    signals: list[str] | None = None,
) -> str:
    """Build the per-diff user message sent to the LLM."""
    signals = signals or []
    payload: dict[str, Any] = {
        "chunk": {
            "index": chunk_index,
            "total": chunk_total,
            "char_start": char_start,
            "char_end": char_end,
        },
        "filename": filename or "",
        "commit_message": commit_message or "",
        # "signals": signals,
        "diff": chunk_text,
    }
    # Use json.dumps for predictable, model-friendly formatting.
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_chat_messages(
    *,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    """Wrap the two prompts into the standard OpenAI-style messages list."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# # Keyword patterns used for signal extraction and for per-category diff-line attribution.
# # Each entry is (signal_name, regex_pattern); signal_name maps to a category via SIGNAL_CATEGORY_MAP.
# SIGNAL_PATTERNS: list[tuple[str, str]] = [
#     ("build_commands",   r"\b(npm run\b|npx\b|make\b|cmake\b|gradle\b|mvn\b|cargo build|go build|xcodebuild\b|swift build|swift run|swift test)\b"),
#     ("runtime_commands", r"\b(docker run|docker-compose|kubectl|npm start|node\s|python -m|uvicorn|gunicorn)\b"),
#     ("tests",            r"\b(npm test|npm run test|test:|tests\b|coverage\b|vitest\b|jest\b|pytest\b)\b"),
#     ("ci_cd",            r"\b(ci/cd|ci-cd|pipeline|github actions|jenkins|circleci|gitlab ci|\.github/workflows)\b"),
#     ("dependencies",     r"\b(dependenc|package\.json|requirements\.txt|pip install|npm install|pnpm add|yarn add)\b"),
#     ("security",         r"\b(security|vulnerability|cve|token|secret|auth|credential)\b"),
#     ("configuration",    r"\b(\.env\b|config\.json|config\.ya?ml|config\.toml|mcp-wordpress\.config\.json)\b"),
#     ("architecture",     r"\b(architecture|design|pattern|interface|dependency injection|composition|inheritance)\b"),
#     ("error_handling",   r"\b(error|exception|retry|timeout|fail(?:ed|ure)?)\b"),
#     ("documentation",    r"\b(readme|claude\.md|agents\.md|docs?|documentation|guide|manual|changelog|jsdoc|docstring|recent updates|version bump(?:ed)?|breaking change)\b"),
#     ("api_contracts",    r"\b(api|endpoint|request|response|http|rest|graphql|rpc)\b"),
#     ("data_handling",    r"\b(schema|serialize|deserialize|parser|validation|zod|json schema)\b"),
#     ("naming",           r"\b(rename|naming|convention|comments?|code style)\b"),
#     ("ui",               r"\b(ui|ux|frontend|component|stylesheet|css|tailwind|accessibility|aria)\b"),
#     ("dev_process",      r"\b(pull request|prs?\b|code review|commit message|conventional commits|git workflow|branching|merge queue|merge request)\b"),
#     ("project_mgmt",     r"\b(backlog|roadmap|milestone|sprint|issue tracker)\b"),
#     ("maintenance",      r"\b(deprecat|backward compat|tech debt|refactor|dead code|duplicates?|maintenance)\b"),
#     ("performance",      r"\b(performance|latency|slow\b|throughput|benchmark|profil|cach(?:ing|e)?\b|optim\w*|memory usage|cpu usage)\b"),
#     ("ai_agent",         r"\b(claude|gpt|copilot|mcp server|agent\b|tool use|prompt|skill\b|slash command)\b"),
#     ("system_overview",  r"\b(project is|library for|framework for|powered by)\b"),
# ]
SIGNAL_PATTERNS: list[tuple[str, str]] = []

# # Signal extraction from diff text, commit message, and filename to provide a soft prior to the LLM
# def build_signals(
#     diff_text: str, filename: str, commit_message: str
# ) -> tuple[list[str], list[str]]:
#     added_text = "\n".join(
#         line[1:]
#         for line in diff_text.splitlines()
#         if line.startswith("+") and not line.startswith("+++")
#     )
#     removed_text = "\n".join(
#         line[1:]
#         for line in diff_text.splitlines()
#         if line.startswith("-") and not line.startswith("---")
#     )
#     lowered_added = f"{commit_message}\n{added_text}".lower()
#     lowered_removed = removed_text.lower()
#     added_sigs: list[str] = []
#     removed_sigs: list[str] = []
#     for name, pattern in SIGNAL_PATTERNS:
#         if re.search(pattern, lowered_added):
#             added_sigs.append(name)
#         elif re.search(pattern, lowered_removed):
#             removed_sigs.append(name)
#     return added_sigs, removed_sigs

def build_signals(*_: str) -> tuple[list[str], list[str]]:
    return [], []


# Build the primary user prompt from the diff and metadata, including signal extraction
def build_primary_prompt(  # pragma: no cover - legacy shim
    *,
    diff_text: str,
    commit_message: str,
    filename: str,
    categories: list[str],
    category_descriptions: dict[str, str],
    signals: list[str],
) -> str:
    """Legacy entry point: returns a user prompt that is still informative."""
    return build_user_prompt(
        chunk_text=diff_text,
        filename=filename,
        commit_message=commit_message,
        chunk_index=0,
        chunk_total=1,
        char_start=0,
        char_end=len(diff_text),
        signals=signals,
    )


def select_category_sample(  # pragma: no cover - legacy shim
    categories: list[str],
    signals: list[str],
    sample_size: int,
) -> list[str]:
    """In multi-label mode we always use the full taxonomy -- no sampling."""
    return list(categories)
