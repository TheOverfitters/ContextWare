from __future__ import annotations

import re
import textwrap

DEFAULT_CATEGORIES = [
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

SIGNAL_CATEGORY_MAP = {
    "build_commands": "Build and Run",
    "runtime_commands": "Build and Run",
    "tests": "Testing",
    "ci_cd": "DevOps",
    "dependencies": "Impl. Details",
    "configuration": "Conf.&Env.",
    "security": "Security",
    "performance": "Performance",
    "documentation": "Documentation",
    "architecture": "Architecture",
    "error_handling": "Debugging",
    "api_contracts": "Impl. Details",
    "data_handling": "Impl. Details",
    "naming": "Impl. Details",
}


def get_primary_prompt_template() -> str:
    template = """\
    You are a code-diff analyst specializing in agent context files (CLAUDE.md, AGENTS.md,
    Copilot instructions, etc.). Given a diff, extract operational rules and key phrases,
    then classify the change into exactly one category.

    
    OUTPUT — return one raw JSON object only (no markdown, no code fences):
    
    {{
    "category": "<name from candidates>",
    "change_type": "addition|modification|deletion|refactor|formatting|metadata|other",
    "key_phrases": ["<short phrase>", ...],
    "rules": ["<imperative rule>", ...],
    "rationale": "<1-2 sentences>",
    "confidence": <float 0.0-1.0>,
    "category_confidences": {{
        "<Candidate A>": <float>,
        "<Candidate B>": <float>
        }}
    }}

    
    FIELD DEFINITIONS
    
    category
    Must match exactly one name from the sampled candidates below.

    change_type — pick the best-fit label:
    addition    New content with no prior equivalent.
    modification Existing content changed in meaning or behavior.
    deletion    Content removed without replacement.
    refactor    Restructured without behavioral change.
    formatting  Whitespace, layout, or punctuation only — no semantic change.
    metadata    Version numbers, dates, status tags, counts, or labels only.
    other       None of the above apply.

    key_phrases
    2-5 short phrases (≤6 words each) a search index could use to retrieve this change.
    Distinct from rules: these describe *what changed*, not *what to do*.

    rules
    0-5 action-oriented, testable imperatives extracted from the added lines.
    Empty array [] when change_type is "formatting" or "metadata".

    rationale
    1-2 sentences justifying the chosen category over the closest runner-up.

    confidence
    Must equal category_confidences[chosen category] exactly.

    category_confidences
    Include every sampled candidate. All values are floats in [0.0, 1.0].
    Values must sum to 1.0.

    
    CLASSIFICATION RULES
    
    1. Base classification on the semantic intent of ADDED lines, not removed ones.
    2. If the diff touches multiple areas, choose the category matching the dominant change.
    3. If documentation embeds operational commands, classify by the commands, not the prose.
    4. Use "System Overview" only when no clear operational guidance exists
    (e.g., pure version bumps, project-summary rewrites).
    5. When uncertain between two categories, lower confidence and spread
    the difference across both in category_confidences.

    
    CONFIDENCE CALIBRATION
    
    0.90-1.00  One clear, unambiguous section matches this category.
    0.70-0.89  Clear match, but the diff also touches secondary areas.
    0.50-0.69  Choosing between two plausible categories.
    0.00-0.49  Guessing — set category to "System Overview" instead.

    
    INPUTS
    
    Candidate categories:
    {categories}

    Auto-extracted signals:
    {signals}

    Diff metadata:
    File   : {filename}
    Message: {commit_message}

    Diff:
    {diff_text}
    """
    return textwrap.dedent(template)


def build_signals(diff_text: str, filename: str, commit_message: str) -> list[str]:
    lowered = f"{filename}\n{commit_message}\n{diff_text}".lower()
    signals: list[str] = []

    def add_signal(name: str, pattern: str) -> None:
        if name not in signals and re.search(pattern, lowered):
            signals.append(name)

    add_signal("build_commands", r"\b(npm run build|npm run docs:generate|make\b|cmake\b|gradle\b|mvn\b|cargo build|go build)\b")
    add_signal("runtime_commands", r"\b(docker run|docker-compose|kubectl|npm start|node\s|python -m|uvicorn|gunicorn)\b")
    add_signal("tests", r"\b(npm test|npm run test|test:|tests\b|coverage\b|vitest\b|jest\b|pytest\b)\b")
    add_signal("ci_cd", r"\b(ci/cd|ci-cd|pipeline|github actions|jenkins|circleci|gitlab ci)\b")
    add_signal("dependencies", r"\b(dependenc|package\.json|requirements\.txt|pip install|npm install|pnpm add|yarn add)\b")
    add_signal("security", r"\b(security|vulnerability|cve|token|secret|auth|credential)\b")
    add_signal("configuration", r"\b(\.env\b|config\.json|config\.ya?ml|config\.toml|mcp-wordpress\.config\.json)\b")
    add_signal("architecture", r"\b(architecture|design|pattern|interface|dependency injection|composition|inheritance)\b")
    add_signal("error_handling", r"\b(error|exception|retry|timeout|fail(?:ed|ure)?)\b")
    add_signal("documentation", r"\b(readme|claude\.md|agents\.md|docs?|documentation|guide|manual)\b")
    add_signal("api_contracts", r"\b(api|endpoint|request|response|http|rest|graphql|rpc)\b")
    add_signal("data_handling", r"\b(schema|serialize|deserialize|parser|validation|zod|json schema)\b")
    add_signal("naming", r"\b(rename|naming|convention)\b")

    return signals


def format_category_descriptions(categories: list[str], descriptions: dict[str, str]) -> str:
    lines = []
    for category in categories:
        description = descriptions.get(category)
        if description:
            lines.append(f"- {category}: {description}")
        else:
            lines.append(f"- {category}")
    return "\n".join(lines) if lines else "-"


def select_category_sample(categories: list[str], signals: list[str], sample_size: int) -> list[str]:
    if sample_size <= 0 or sample_size >= len(categories):
        return categories

    selected: list[str] = []
    for signal in signals:
        mapped = SIGNAL_CATEGORY_MAP.get(signal)
        if mapped and mapped in categories and mapped not in selected:
            selected.append(mapped)

    if "System Overview" in categories and "System Overview" not in selected:
        selected.insert(0, "System Overview")

    for category in categories:
        if len(selected) >= sample_size:
            break
        if category not in selected:
            selected.append(category)

    return selected


def build_primary_prompt(
    *,
    diff_text: str,
    commit_message: str,
    filename: str,
    categories: list[str],
    category_descriptions: dict[str, str],
    signals: list[str],
) -> str:
    categories_text = format_category_descriptions(categories, category_descriptions)
    signals_text = "\n".join(f"- {signal}" for signal in signals) if signals else "-"
    template = get_primary_prompt_template()
    return template.format(
        categories=categories_text,
        signals=signals_text,
        commit_message=commit_message or "",
        filename=filename or "",
        diff_text=diff_text,
    )