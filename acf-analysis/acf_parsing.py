from __future__ import annotations

import json
import re
from typing import Any

# from acf_prompt import SIGNAL_CATEGORY_MAP


# Repairs a JSON snippet that failed to parse, If the snippet is not reparable, returns None.
def _repair_json_snippet(snippet: str) -> dict[str, Any] | None:
    """Try common repairs on a JSON snippet that failed json.loads.

    Handles:
      * stray all-whitespace quoted strings the model emits as botched
        indentation, e.g.  ..."System Overview","     "summary":...  where
        the next key's opening quote was merged into a whitespace string;
      * trailing commas before } or ];
      * truncated structures where the model stopped generating mid-object.
    """
    # Collapse  "<whitespace>"<word>"  ->  "<word>"  (the de-indentation tic).
    snippet = re.sub(r'"\s+"(\w+)"', r'"\1"', snippet)
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r",\s*([}\]])", r"\1", snippet)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    open_braces = fixed.count("{") - fixed.count("}")
    open_brackets = fixed.count("[") - fixed.count("]")
    if open_braces > 0 or open_brackets > 0:
        last_sep = max(fixed.rfind(","), fixed.rfind("["), 0)
        if last_sep > 0:
            closed = fixed[:last_sep] + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
            try:
                return json.loads(closed)
            except json.JSONDecodeError:
                pass
    return None

# Extract category scores from raw text using regex patterns, even when JSON parsing fails
def _extract_partial_vector(raw_text: str, categories: list[str]) -> dict[str, float]:
    vector: dict[str, float] = {c: 0.0 for c in categories}

    # Pattern 1: inline key-value  "Category": 0.8
    for category in categories:
        m = re.search(
            r'["\']?' + re.escape(category) + r'["\']?\s*[:\s]+\s*([0-9]+(?:\.[0-9]+)?)',
            raw_text,
            re.IGNORECASE,
        )
        if m:
            try:
                vector[category] = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass

    # Pattern 2 & 3: {"name": "Category", ..., "score": 0.8} within 300 chars
    for category in categories:
        if vector[category] > 0.0:
            continue
        for pat in (
            r'"name"\s*:\s*["\']' + re.escape(category) + r'["\'].{0,300}?"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?).{0,300}?"name"\s*:\s*["\']' + re.escape(category) + r'["\']',
        ):
            m = re.search(pat, raw_text, re.IGNORECASE | re.DOTALL)
            if m:
                try:
                    vector[category] = max(0.0, min(1.0, float(m.group(1))))
                    break
                except ValueError:
                    pass

    # Pattern 4: primary_category field → boost that category if still zero
    m = re.search(r'"primary_category"\s*:\s*["\']([^"\']+)["\']', raw_text, re.IGNORECASE)
    if m:
        primary = m.group(1).strip()
        if primary in vector and vector[primary] == 0.0:
            vector[primary] = 0.65

    return vector


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Find and return the first balanced top-level JSON object in *raw_text*.

    Handles extra prose/markdown around the response and is escape-aware
    inside string literals. When json.loads fails on a balanced snippet,
    attempts JSON repair before moving on.
    """
    raw_text = re.sub(r'"\s+"(\w+)"', r'"\1"', raw_text)
    raw_text = re.sub(r'}(\s*,\s*)("primary_category")', r'}]\1\2', raw_text)

    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(raw_text):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue

        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                snippet = raw_text[start : idx + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    repaired = _repair_json_snippet(snippet)
                    if repaired is not None:
                        return repaired
                    start = None
    return None


# Score coercion and validation helpers for multi-label parsing.
def _coerce_score(value: Any) -> float | None:
    """Coerce *value* to a float in [0.0, 1.0]; return None on failure.

    Tolerates the model emitting a score as a numeric string (e.g.
    ``"score": "0.85"`` or ``"0,85"``) instead of a JSON number, which would
    otherwise be silently dropped to 0.0.
    """
    if isinstance(value, bool):
        # bool is a subclass of int -- treat explicitly.
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        # Accept a numeric string, tolerating a comma decimal separator and
        # surrounding whitespace/quotes the model sometimes leaves in.
        cleaned = value.strip().strip('"').strip("'").replace(",", ".")
        try:
            return max(0.0, min(1.0, float(cleaned)))
        except ValueError:
            return None
    return None


def validate_multilabel_payload(
    payload: dict[str, Any],
    categories: list[str],
) -> list[str]:
    """Return a list of human-readable validation errors (empty == OK).

    A well-formed payload has:
      * ``categories`` -- a list with one entry per category name; each
        entry has ``name``, ``score`` (0.0-1.0), and ``rationale``.
      * ``primary_category`` -- one of the known category names.
      * ``summary`` -- a short non-empty string.
    """
    errors: list[str] = []
    cat_entries = payload.get("categories")
    if not isinstance(cat_entries, list):
        return ["categories_missing_or_not_list"]

    seen: set[str] = set()
    for idx, entry in enumerate(cat_entries):
        if not isinstance(entry, dict):
            errors.append(f"categories[{idx}].not_object")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"categories[{idx}].name_invalid")
            continue
        seen.add(name)
        score = _coerce_score(entry.get("score"))
        if score is None:
            errors.append(f"categories[{idx}].score_invalid({name})")
        rationale = entry.get("rationale")
        if not isinstance(rationale, str):
            errors.append(f"categories[{idx}].rationale_invalid({name})")

    missing = [c for c in categories if c not in seen]
    if missing:
        errors.append(f"categories_missing_entries:{','.join(missing)}")

    primary = payload.get("primary_category")
    if not isinstance(primary, str) or primary not in categories:
        errors.append("primary_category_invalid")

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary_invalid")

    return errors

# Multilabel parsing and aggregation helpers, including rescue heuristics when parsing fails.
def parse_multilabel_response(
    raw_text: str,
    categories: list[str],
) -> tuple[dict[str, Any] | None, list[str], dict[str, float]]:
    """Parse a multi-label LLM response.

    Returns ``(parsed, errors, score_vector)`` where:
      * ``parsed`` is the extracted JSON object (or None on extraction
        failure).
      * ``errors`` is the list of validation issues (empty == OK).
      * ``score_vector`` is ``{category: float}`` derived from the
        payload's ``categories`` array, restricted to *categories*
        and clamped to [0.0, 1.0]. When validation fails or
        extraction fails, an all-zero vector is returned.
    """
    zero_vector = {c: 0.0 for c in categories}
    parsed = extract_json_object(raw_text)
    if parsed is None:
        partial = _extract_partial_vector(raw_text, categories)
        rescue_vector = partial if any(v > 0.0 for v in partial.values()) else zero_vector
        return None, ["invalid_json"], rescue_vector

    errors = validate_multilabel_payload(parsed, categories)
    vector = _vector_from_payload(parsed, categories)
    return parsed, errors, vector


# def signals_fallback_vector(signals: list[str], categories: list[str]) -> dict[str, float]:
#     vector = {c: 0.0 for c in categories}
#     for signal in signals:
#         mapped = SIGNAL_CATEGORY_MAP.get(signal)
#         if mapped and mapped in vector:
#             current = vector[mapped]
#             vector[mapped] = min(0.90, current + 0.50 * (1.0 - current))
#     return vector


def _vector_from_payload(
    payload: dict[str, Any],
    categories: list[str],
    key: str = "categories",
) -> dict[str, float]:
    """Build a clean {name: score} dict from a validated payload.

    Reads the array under *key* (``"categories"`` for the ACF topic axis,
    ``"maintenance_types"`` for the maintenance-reason axis). Tries exact
    match first, then case-insensitive fallback so minor capitalisation
    differences from the model do not zero out the vector.
    """
    vector: dict[str, float] = {c: 0.0 for c in categories}
    lower_map = {c.lower(): c for c in categories}
    for entry in payload.get(key, []) or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        canonical = name if name in vector else lower_map.get(name.lower())
        if canonical is None:
            continue
        score = _coerce_score(entry.get("score"))
        if score is not None:
            vector[canonical] = score
    return vector


# Maintenance-reason axis (ISO/IEC/IEEE 14764) parsing. Kept separate from the
# category retry logic: it reads the same parsed object but never triggers a
# retry on its own -- when the maintenance block is missing we return zeros.
def maintenance_vector_from_payload(
    parsed: dict[str, Any] | None,
    maintenance_types: list[str],
) -> dict[str, float]:
    """Return {maintenance_type: score} from a parsed multi-label payload."""
    if not isinstance(parsed, dict):
        return {t: 0.0 for t in maintenance_types}
    return _vector_from_payload(parsed, maintenance_types, key="maintenance_types")


# Aggregation helper that re-exports the one from acf_chunker so callers don't need to depend on that module directly.
def aggregate_chunk_vectors(
    chunk_vectors: list[dict[str, float] | None],
    categories: list[str],
    aggregation: str = "max",
) -> dict[str, float]:
    """Combine per-chunk vectors into a final per-diff vector.

    Thin wrapper that imports :func:`acf_chunker.aggregate_chunk_classifications`
    so callers only need to depend on this module.
    """
    from acf_chunker import aggregate_chunk_classifications  # local import

    return aggregate_chunk_classifications(chunk_vectors, categories, aggregation)


# Allowed change types for the "change_type" field in the primary classification output.
ALLOWED_CHANGE_TYPES = {
    "addition",
    "modification",
    "deletion",
    "refactor",
    "formatting",
    "metadata",
    "other",
}

# Primary classification parsing and validation, including confidence gating and fallback construction.
def build_fallback_primary(raw_text: str, fallback_category: str) -> dict[str, Any]:
    return {
        "category": fallback_category,
        "change_type": "other",
        "key_phrases": [],
        "rules": [],
        "rationale": "Model response was not valid JSON; fallback applied.",
        "confidence": 0.0,
        "raw": raw_text,
    }

# Validation and confidence gating for the primary classification output.
def validate_primary_payload(payload: dict[str, Any], categories: list[str]) -> list[str]:
    errors: list[str] = []
    if payload.get("category") not in categories:
        errors.append("category_missing_or_invalid")
    if payload.get("change_type") not in ALLOWED_CHANGE_TYPES:
        errors.append("change_type_missing_or_invalid")
    key_phrases = payload.get("key_phrases")
    if not isinstance(key_phrases, list) or not all(isinstance(item, str) for item in key_phrases):
        errors.append("key_phrases_invalid")
    rules = payload.get("rules")
    if not isinstance(rules, list) or not all(isinstance(item, str) for item in rules):
        errors.append("rules_invalid")
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("rationale_invalid")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)):
        errors.append("confidence_invalid")
    category_confidences = payload.get("category_confidences")
    if category_confidences is not None and not isinstance(category_confidences, dict):
        errors.append("category_confidences_invalid")
    return errors

# Confidence gating for the primary classification output, with fallback to a default category when confidence is too low.
def apply_confidence_gate(
    parsed: dict[str, Any],
    fallback_category: str,
    threshold: float = 0.50,
) -> dict[str, Any]:
    confidence = parsed.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    if confidence == 0.0:
        category_confidences = parsed.get("category_confidences")
        if isinstance(category_confidences, dict):
            category = parsed.get("category")
            if isinstance(category, str):
                mapped_confidence = category_confidences.get(category)
                if isinstance(mapped_confidence, (int, float)):
                    confidence = float(mapped_confidence)
                    parsed["confidence"] = confidence

    if confidence < threshold:
        parsed["rationale"] = (
            f"[confidence-gate] Original category '{parsed.get('category')}' "
            f"had confidence {confidence:.2f} < {threshold}. "
            f"Reassigned to fallback. Original rationale: {parsed.get('rationale', '')}"
        )
        parsed["category"] = fallback_category
        parsed["confidence"] = confidence

    return parsed

# Normalization of category confidence vectors to ensure they sum to 1.0, and removal of invalid entries.
def normalize_category_confidences(
    parsed: dict[str, Any],
    categories: list[str],
) -> dict[str, Any]:
    raw_confidences = parsed.get("category_confidences")
    if not isinstance(raw_confidences, dict):
        return parsed

    cleaned: dict[str, float] = {}
    total = 0.0
    for category in categories:
        raw_value = raw_confidences.get(category)
        if isinstance(raw_value, (int, float)):
            value = max(0.0, float(raw_value))
            cleaned[category] = value
            total += value

    if not cleaned:
        return parsed

    if total > 0:
        for category in cleaned:
            cleaned[category] = cleaned[category] / total

    parsed["category_confidences"] = cleaned
    return parsed

# Main entry point for parsing the primary classification response, including JSON extraction, validation, confidence gating, and fallback construction.
def parse_primary_response(
    raw_text: str,
    categories: list[str],
    fallback_category: str,
    threshold: float,
    confidence_categories: list[str] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    parsed = extract_json_object(raw_text)
    if parsed is None:
        return None, ["invalid_json"]
    validation_errors = validate_primary_payload(parsed, categories)
    if validation_errors:
        return parsed, validation_errors
    parsed = apply_confidence_gate(
        parsed,
        fallback_category=fallback_category,
        threshold=threshold,
    )
    return parsed, []

# Build a category confidence vector from the parsed primary classification output, using the "category_confidences" field if present, or falling back to a heuristic based on signals.
def build_category_confidences(
    categories: list[str],
    parsed: dict[str, Any] | None,
    # signals: list[str],
) -> dict[str, float]:
    if not categories:
        return {}

    if not parsed:
        uniform = 1.0 / len(categories)
        return {category: uniform for category in categories}

    raw_confidences = parsed.get("category_confidences")
    if isinstance(raw_confidences, dict):
        result: dict[str, float] = {}
        for category in categories:
            raw_value = raw_confidences.get(category)
            if isinstance(raw_value, (int, float)):
                result[category] = max(0.0, min(1.0, float(raw_value)))
            else:
                result[category] = 0.0
        return result

    category = parsed.get("category")
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0

    confidence = max(0.0, min(1.0, float(confidence)))
    confidences = {cat: 0.0 for cat in categories}

    if isinstance(category, str) and category in confidences:
        confidences[category] = confidence
        remaining = 1.0 - confidence
    else:
        remaining = 1.0

    weights: dict[str, float] = {cat: 1.0 for cat in categories if cat != category}
    # for signal in signals:
    #     mapped = SIGNAL_CATEGORY_MAP.get(signal)
    #     if mapped in weights:
    #         weights[mapped] += 2.0

    total_weight = sum(weights.values())
    if total_weight <= 0:
        uniform = remaining / max(1, len(weights))
        for cat in weights:
            confidences[cat] = uniform
    else:
        for cat, weight in weights.items():
            confidences[cat] = remaining * (weight / total_weight)

    return confidences
