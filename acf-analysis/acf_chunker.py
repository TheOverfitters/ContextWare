from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any



_HUNK_HEADER = re.compile(r"^@@ .* @@")

_FILE_HEADER = re.compile(r"^diff --git ")


@dataclass
class DiffChunk:
    """A single chunk of a diff, ready to be sent to the LLM."""

    index: int
    total: int
    text: str
    char_start: int
    char_end: int

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.index == self.total - 1


@dataclass
class ChunkPlan:
    """Result of splitting a diff into chunks."""

    chunks: list[DiffChunk]
    truncated: bool
    original_chars: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "truncated": self.truncated,
            "original_chars": self.original_chars,
            "chunk_count": len(self.chunks),
            "chunk_indices": [c.index for c in self.chunks],
        }


def plan_chunks(
    diff_text: str,
    max_chars: int,
    overlap_chars: int = 400,
    prefer_hunk_boundaries: bool = True,
) -> ChunkPlan:
    """Split a diff into DiffChunk objects that fit within ``max_chars``.

    Strategy:
      1. If the diff fits in ``max_chars``, return it as a single chunk.
      2. Otherwise, walk the diff line by line and greedily accumulate
         up to ``max_chars`` characters, preferring to break at file
         (``diff --git ...``) or hunk (``@@ ... @@``) boundaries.
      3. Each new chunk overlaps the previous one by ``overlap_chars``
         so context is not lost between splits.

    Args:
        diff_text: Raw unified diff text.
        max_chars: Hard upper bound on chunk size (including overlap).
        overlap_chars: Number of trailing characters to repeat at the
            start of the next chunk to preserve boundary context.
        prefer_hunk_boundaries: When True, try to end a chunk on a
            file/hunk header line.
    """
    if not diff_text:
        return ChunkPlan(chunks=[], truncated=False, original_chars=0)

    original_chars = len(diff_text)
    if original_chars <= max_chars:
        chunk = DiffChunk(
            index=0,
            total=1,
            text=diff_text,
            char_start=0,
            char_end=original_chars,
        )
        return ChunkPlan(chunks=[chunk], truncated=False, original_chars=original_chars)

    if overlap_chars < 0:
        overlap_chars = 0
    if overlap_chars >= max_chars:
        overlap_chars = max(0, max_chars // 4)

    lines = diff_text.splitlines(keepends=True)
    chunks: list[DiffChunk] = []
    current_start = 0
    current_len = 0
    last_boundary_end: int | None = None

    def close_chunk(end: int) -> DiffChunk:
        text = diff_text[current_start:end]
        return DiffChunk(
            index=len(chunks),
            total=0,  # patched after the loop
            text=text,
            char_start=current_start,
            char_end=end,
        )

    for line in lines:
        line_len = len(line)
        # Decide if this line is a good cut point.
        is_boundary = (
            _FILE_HEADER.match(line) is not None
            if prefer_hunk_boundaries
            else _HUNK_HEADER.match(line) is not None
        )
        if is_boundary:
            last_boundary_end = current_start + current_len

        # Would adding this line exceed the budget?
        if current_len + line_len > max_chars and current_len > 0:
            # Close current chunk. Prefer the last good boundary if it
            # keeps the chunk reasonably sized.
            cut_at = current_start + current_len
            if (
                last_boundary_end is not None
                and last_boundary_end > current_start
                and (last_boundary_end - current_start) >= max_chars // 2
            ):
                cut_at = last_boundary_end

            chunks.append(close_chunk(cut_at))
            # Start next chunk with overlap from the cut point.
            overlap_start = max(0, cut_at - overlap_chars)
            current_start = overlap_start
            current_len = cut_at - overlap_start
            last_boundary_end = None

        current_len += line_len

    # Tail chunk.
    if current_len > 0:
        chunks.append(close_chunk(current_start + current_len))

    # Patch total counts and indices.
    for i, c in enumerate(chunks):
        c.index = i
        c.total = len(chunks)

    truncated = original_chars > sum(len(c.text) for c in chunks)
    return ChunkPlan(chunks=chunks, truncated=truncated, original_chars=original_chars)


def aggregate_chunk_classifications(
    chunk_results: list[dict[str, float] | None],
    categories: list[str],
    aggregation: str = "max",
) -> dict[str, float]:
    """Combine per-chunk classification vectors into a single one.

    Each ``chunk_results[i]`` is a {category: confidence} dict (or None
    if the chunk failed to classify). Aggregation modes:
      * ``max``     – keep the maximum confidence seen for each category
                      (good for "is this category present at all?").
      * ``mean``    – average across successful chunks.
      * ``any``     – 1.0 if any chunk has confidence >= 0.5, else 0.0.

    Missing chunks and missing categories default to 0.0.
    """
    if aggregation not in {"max", "mean", "any"}:
        raise ValueError(f"Unknown aggregation mode: {aggregation!r}")

    valid = [r for r in chunk_results if isinstance(r, dict)]
    if not valid:
        return {c: 0.0 for c in categories}

    result: dict[str, float] = {}
    for cat in categories:
        values = [r.get(cat, 0.0) for r in valid]
        if not values:
            result[cat] = 0.0
            continue
        if aggregation == "max":
            result[cat] = max(values)
        elif aggregation == "mean":
            result[cat] = sum(values) / len(valid)
        else:  # any
            result[cat] = 1.0 if any(v >= 0.5 for v in values) else 0.0
    return result
