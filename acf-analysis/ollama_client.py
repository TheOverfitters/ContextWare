from __future__ import annotations

import json
import logging
import random
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_KEY_PATTERN = re.compile(r"ollama_[A-Za-z0-9]+")

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Models for which we've already announced the empty-content -> thinking/reasoning
# fallback. For reasoning models (e.g. qwen3.5:cloud) this fallback is the *normal*
# code path and fires on every request, so we log it once per (model, field) at
# INFO and drop subsequent occurrences to DEBUG to keep multi-model runs readable.
_THINKING_FALLBACK_SEEN: set[tuple[str, str]] = set()


@dataclass
class OllamaChatResult:
    content: str
    logprobs: list[float] | None


def redact_secrets(text: str, known_keys: list[str] | None = None) -> str:
    sanitized = text
    if known_keys:
        for key in known_keys:
            if key:
                sanitized = sanitized.replace(key, "[REDACTED_OLLAMA_KEY]")
    return OLLAMA_KEY_PATTERN.sub("[REDACTED_OLLAMA_KEY]", sanitized)


def parse_retry_after_seconds(text: str) -> float | None:
    match = re.search(r"(?:in\s+)?(\d+(?:\.\d+)?)\s*s", text.lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def get_retry_after_from_headers(headers: dict[str, str]) -> float | None:
    for header_name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        raw_value = headers.get(header_name)
        if not raw_value:
            continue
        match = re.search(r"\d+(?:\.\d+)?", raw_value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                continue
    return None


class SlidingRateLimiter:
    def __init__(self, requests_per_minute: int, tokens_per_minute: int, min_interval_seconds: float):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.min_interval_seconds = min_interval_seconds
        self.request_timestamps: list[float] = []
        self.token_events: list[tuple[float, int]] = []
        self.last_request_ts = 0.0

    def _cleanup_windows(self, now_ts: float) -> int:
        self.request_timestamps = [ts for ts in self.request_timestamps if now_ts - ts < 60]
        self.token_events = [event for event in self.token_events if now_ts - event[0] < 60]
        return sum(tokens for _, tokens in self.token_events)

    def wait_for_budget(self, estimated_tokens_needed: int) -> None:
        while True:
            now_ts = time.time()
            current_tokens = self._cleanup_windows(now_ts)

            req_wait = 0.0
            if len(self.request_timestamps) >= self.requests_per_minute:
                req_wait = max(0.0, 60 - (now_ts - self.request_timestamps[0]) + 0.05)

            token_wait = 0.0
            if current_tokens + estimated_tokens_needed > self.tokens_per_minute:
                overflow = current_tokens + estimated_tokens_needed - self.tokens_per_minute
                released = 0
                for ts, tokens in self.token_events:
                    released += tokens
                    if released >= overflow:
                        token_wait = max(0.0, 60 - (now_ts - ts) + 0.05)
                        break

            pacing_wait = max(0.0, self.min_interval_seconds - (now_ts - self.last_request_ts))
            wait_seconds = max(req_wait, token_wait, pacing_wait)

            if wait_seconds <= 0:
                return

            logger.info(
                "THROTTLE | waiting %.1fs (rpm=%s/%s, tpm~=%s/%s)",
                wait_seconds,
                len(self.request_timestamps),
                self.requests_per_minute,
                current_tokens,
                self.tokens_per_minute,
            )
            time.sleep(wait_seconds)

    def register_request(self, estimated_tokens: int) -> None:
        now_ts = time.time()
        self.request_timestamps.append(now_ts)
        self.token_events.append((now_ts, estimated_tokens))
        self.last_request_ts = now_ts


def fetch_model_context_size(
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: int = 10,
) -> int | None:
    """Query Ollama /api/show to read the model's configured context size.

    Tries, in order:
      1. ``model_info`` dict (any key containing "context_length") — newer Ollama
      2. ``parameters`` field parsed for ``num_ctx <value>``
      3. ``modelfile`` parsed for ``PARAMETER num_ctx <value>``

    Returns None if the endpoint is unavailable or the value cannot be found.
    """
    endpoint = f"{base_url.rstrip('/')}/api/show"
    payload = json.dumps({"model": model}).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None

    # Newer Ollama exposes model_info with architecture-prefixed keys.
    model_info = data.get("model_info") or {}
    for key, val in model_info.items():
        if "context_length" in key and isinstance(val, int) and val > 0:
            return val

    # Older Ollama exposes a plain "parameters" string.
    parameters = data.get("parameters", "")
    if isinstance(parameters, str):
        m = re.search(r"num_ctx\s+(\d+)", parameters)
        if m:
            return int(m.group(1))

    # Fallback: parse the modelfile directly.
    modelfile = data.get("modelfile", "")
    if isinstance(modelfile, str):
        m = re.search(r"PARAMETER\s+num_ctx\s+(\d+)", modelfile, re.IGNORECASE)
        if m:
            return int(m.group(1))

    return None


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def _coerce_logprob_list(values: list[Any]) -> list[float] | None:
    logprobs: list[float] = []
    for item in values:
        if isinstance(item, (int, float)):
            logprobs.append(float(item))
            continue
        if isinstance(item, dict):
            for key in ("logprob", "log_prob", "token_logprob"):
                raw = item.get(key)
                if isinstance(raw, (int, float)):
                    logprobs.append(float(raw))
                    break
    return logprobs or None


def _coerce_logprobs(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return _coerce_logprob_list(value)
    if isinstance(value, dict):
        for key in ("token_logprobs", "token_log_probs", "logprobs", "log_probs"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                coerced = _coerce_logprob_list(candidate)
                if coerced:
                    return coerced
    return None


def extract_logprobs(parsed: dict[str, Any], prefer_prompt: bool) -> list[float] | None:
    candidates: list[Any] = []
    if prefer_prompt:
        candidates.extend(
            [
                parsed.get("prompt_logprobs"),
                parsed.get("prompt_logprob"),
                parsed.get("prompt_log_probs"),
            ]
        )
    candidates.extend(
        [
            parsed.get("logprobs"),
            parsed.get("response_logprobs"),
            parsed.get("response_logprob"),
            parsed.get("eval_logprobs"),
        ]
    )

    message = parsed.get("message")
    if isinstance(message, dict):
        candidates.extend(
            [
                message.get("logprobs"),
                message.get("response_logprobs"),
                message.get("response_logprob"),
            ]
        )

    for candidate in candidates:
        coerced = _coerce_logprobs(candidate)
        if coerced:
            return coerced
    return None


def _add_jitter(wait_seconds: float) -> float:
    if wait_seconds <= 0:
        return wait_seconds
    jitter = random.uniform(0.0, min(1.0, wait_seconds * 0.1))
    return wait_seconds + jitter


def _call_with_retries(
    *,
    request_fn: Callable[[], T],
    rate_limiter: "SlidingRateLimiter",
    estimated_tokens: int,
    retry_per_prompt: int,
    rate_limit_wait_base_seconds: int,
    rate_limit_wait_max_seconds: int,
    hard_cooldown_seconds: int,
    api_key: str,
) -> T:
    rate_limit_retries = 0
    server_retries = 0
    network_retries = 0

    while True:
        rate_limiter.wait_for_budget(estimated_tokens)
        rate_limiter.register_request(estimated_tokens)

        try:
            return request_fn()
        except HTTPError as err:
            try:
                err_body = err.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            err_text = f"status={err.code} {err.reason} {err_body}".strip()

            if err.code == 429:
                rate_limit_retries += 1
                headers = {k.lower(): v for k, v in err.headers.items()}
                retry_after = get_retry_after_from_headers(headers)
                if retry_after is None:
                    retry_after = parse_retry_after_seconds(err_text)

                adaptive_wait = min(
                    rate_limit_wait_base_seconds * max(1, rate_limit_retries),
                    rate_limit_wait_max_seconds,
                )
                wait_seconds = min(
                    rate_limit_wait_max_seconds,
                    max(retry_after if retry_after is not None else 0.0, adaptive_wait),
                )
                if rate_limit_retries >= retry_per_prompt:
                    wait_seconds = max(wait_seconds, hard_cooldown_seconds)
                    rate_limit_retries = 0

                wait_seconds = _add_jitter(wait_seconds)
                logger.warning(
                    "RATE_LIMIT | retry_after_api=%ss | retrying in %.1fs",
                    retry_after if retry_after is not None else "n/a",
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            if err.code in {401, 403}:
                raise RuntimeError("Ollama authentication failed. Check OLLAMA_API_KEY and base URL.") from err

            if err.code >= 500:
                if server_retries < retry_per_prompt:
                    server_retries += 1
                    wait_time = _add_jitter(min(15, server_retries * 2))
                    logger.warning(
                        "SERVER_RETRY(%s) | retry %s/%s in %.1fs",
                        err.code,
                        server_retries,
                        retry_per_prompt,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue

            raise RuntimeError(f"Unrecoverable API error: {redact_secrets(err_text, [api_key])}") from err
        except (URLError, TimeoutError, socket.timeout, ConnectionError) as err:
            if network_retries < retry_per_prompt:
                network_retries += 1
                wait_time = _add_jitter(min(15, network_retries * 2))
                logger.warning(
                    "NETWORK_RETRY | retry %s/%s in %.1fs",
                    network_retries,
                    retry_per_prompt,
                    wait_time,
                )
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"Unrecoverable network error: {redact_secrets(str(err), [api_key])}") from err
        except (json.JSONDecodeError, ValueError) as err:
            raise RuntimeError(f"Invalid API response: {redact_secrets(str(err), [api_key])}") from err


def call_ollama_chat_completion(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_completion_tokens: int,
    top_p: float,
    request_logprobs: bool,
    response_schema: dict | None = None,
    context_size: int | None = None,
) -> OllamaChatResult:
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_completion_tokens,
    }
    if context_size is not None:
        options["num_ctx"] = context_size
    if request_logprobs:
        options["logprobs"] = True

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
        "logprobs": request_logprobs,
    }
    if response_schema is not None:
        payload["format"] = response_schema

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urlopen(request, timeout=timeout_seconds) as response:
        raw_response = response.read().decode("utf-8", errors="replace")

    parsed = json.loads(raw_response)
    message = parsed.get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing 'message' field in Ollama response")

    content = message.get("content", "")

    # Some reasoning/thinking models produce empty content and put the actual
    # answer in a separate field (e.g. "thinking", "reasoning_content").
    # Fall back to those fields so the pipeline can still extract JSON.
    if not content:
        for alt_key in ("thinking", "reasoning_content", "reasoning", "tool_calls"):
            alt = message.get(alt_key)
            if isinstance(alt, str) and alt.strip():
                seen_key = (model, alt_key)
                level = logging.DEBUG if seen_key in _THINKING_FALLBACK_SEEN else logging.INFO
                _THINKING_FALLBACK_SEEN.add(seen_key)
                logger.log(
                    level,
                    "EMPTY_CONTENT | %s: message.content empty; using message.%s "
                    "(%d chars). Normal for reasoning models; logged once per model.",
                    model,
                    alt_key,
                    len(alt),
                )
                content = alt
                break
            if alt_key == "tool_calls" and isinstance(alt, list) and alt:
                # Some models encode the JSON answer as a tool-call argument.
                try:
                    tc_args = alt[0].get("function", {}).get("arguments", "")
                    if tc_args:
                        seen_key = (model, "tool_calls")
                        level = logging.DEBUG if seen_key in _THINKING_FALLBACK_SEEN else logging.INFO
                        _THINKING_FALLBACK_SEEN.add(seen_key)
                        logger.log(
                            level,
                            "EMPTY_CONTENT | %s: using tool_calls[0].function.arguments. "
                            "Normal for reasoning models; logged once per model.",
                            model,
                        )
                        content = tc_args
                        break
                except Exception:
                    pass

    if not content:
        # Genuinely problematic: empty content AND no usable fallback field.
        # Always warn and dump the raw response so the operator can diagnose.
        logger.warning(
            "EMPTY_CONTENT | %s: message.content is empty and no fallback found. "
            "Full response (first 500 chars): %s",
            model,
            raw_response[:500],
        )

    logprobs = extract_logprobs(parsed, prefer_prompt=False) if request_logprobs else None
    return OllamaChatResult(content=str(content), logprobs=logprobs)


def call_ollama_generate_logprobs(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    num_predict: int,
) -> list[float] | None:
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    options = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": num_predict,
        "logprobs": True,
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
        "logprobs": True,
    }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urlopen(request, timeout=timeout_seconds) as response:
        raw_response = response.read().decode("utf-8", errors="replace")

    parsed = json.loads(raw_response)
    return extract_logprobs(parsed, prefer_prompt=True)


def call_chat_with_retries(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_completion_tokens: int,
    rate_limiter: SlidingRateLimiter,
    retry_per_prompt: int,
    rate_limit_wait_base_seconds: int,
    rate_limit_wait_max_seconds: int,
    hard_cooldown_seconds: int,
    request_logprobs: bool,
    response_schema: dict | None = None,
    context_size: int | None = None,
) -> OllamaChatResult:
    estimated_tokens_needed = (
        sum(estimate_tokens(m.get("content", "")) for m in messages)
        + max_completion_tokens + 500
    )
    return _call_with_retries(
        request_fn=lambda: call_ollama_chat_completion(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            messages=messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            request_logprobs=request_logprobs,
            response_schema=response_schema,
            context_size=context_size,
        ),
        rate_limiter=rate_limiter,
        estimated_tokens=estimated_tokens_needed,
        retry_per_prompt=retry_per_prompt,
        rate_limit_wait_base_seconds=rate_limit_wait_base_seconds,
        rate_limit_wait_max_seconds=rate_limit_wait_max_seconds,
        hard_cooldown_seconds=hard_cooldown_seconds,
        api_key=api_key,
    )


def call_generate_logprobs_with_retries(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    num_predict: int,
    rate_limiter: SlidingRateLimiter,
    retry_per_prompt: int,
    rate_limit_wait_base_seconds: int,
    rate_limit_wait_max_seconds: int,
    hard_cooldown_seconds: int,
) -> list[float] | None:
    estimated_tokens_needed = estimate_tokens(prompt) + max(1, num_predict) + 200
    return _call_with_retries(
        request_fn=lambda: call_ollama_generate_logprobs(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            prompt=prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict,
        ),
        rate_limiter=rate_limiter,
        estimated_tokens=estimated_tokens_needed,
        retry_per_prompt=retry_per_prompt,
        rate_limit_wait_base_seconds=rate_limit_wait_base_seconds,
        rate_limit_wait_max_seconds=rate_limit_wait_max_seconds,
        hard_cooldown_seconds=hard_cooldown_seconds,
        api_key=api_key,
    )
