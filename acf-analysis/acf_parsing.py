from __future__ import annotations

import json
from typing import Any

from acf_prompt import SIGNAL_CATEGORY_MAP


ALLOWED_CHANGE_TYPES = {
    "addition",
    "modification",
    "deletion",
    "refactor",
    "formatting",
    "metadata",
    "other",
}


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
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
                    start = None
    return None


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


def validate_primary_payload(payload: dict[str, Any], categories: list[str]) -> list[str]:
    errors = []
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


def build_category_confidences(
    categories: list[str],
    parsed: dict[str, Any] | None,
    signals: list[str],
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
    for signal in signals:
        mapped = SIGNAL_CATEGORY_MAP.get(signal)
        if mapped in weights:
            weights[mapped] += 2.0

    total_weight = sum(weights.values())
    if total_weight <= 0:
        uniform = remaining / max(1, len(weights))
        for cat in weights:
            confidences[cat] = uniform
    else:
        for cat, weight in weights.items():
            confidences[cat] = remaining * (weight / total_weight)

    return confidences