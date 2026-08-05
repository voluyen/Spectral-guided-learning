"""Step segmentation of long CoT responses with exact token-offset boundaries.

The paper specifies "standardized step-wise segmentation" without details; we split at
logical boundaries — after sentence-ending punctuation (or a closing brace/bracket, which
ends inline LaTeX) followed by whitespace and a capital letter. The split is deterministic
and preserves semantically complete steps. No post-hoc merging is applied: every boundary
found by the rule becomes a step, so segmentation depends only on the text itself.
"""

import re

_STEP_BOUNDARY = re.compile(r"([.?!\}\]])([\s\n]+)([A-Z])")


def split_into_step_texts(response: str) -> list[str]:
    """Split a response into step texts at sentence-level logical boundaries.

    Concatenating the returned pieces reproduces the original string exactly, so token
    offsets computed from cumulative prefixes stay aligned with the tokenized response.
    """
    if not response:
        return []

    pieces, cursor = [], 0
    for match in _STEP_BOUNDARY.finditer(response):
        boundary = match.start(3)  # cut before the capital letter, keep whitespace behind
        pieces.append(response[cursor:boundary])
        cursor = boundary
    if cursor < len(response):
        pieces.append(response[cursor:])
    return pieces


def record_step_spans(record: dict) -> list[tuple[int, int]]:
    """Absolute (start, end) token spans of a data_prep.py record's steps."""
    return [(step["token_start"], step["token_end"]) for step in record["steps"]]


def segment_response_token_spans(
    tokenizer,
    prompt_ids: list[int],
    response: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize a response and map each step to an absolute token span.

    Boundaries are derived from incremental prefix tokenization, which is exact for
    BPE tokenizers as long as steps are concatenated back in order. Zero-token steps
    (possible when a piece tokenizes into the previous piece's trailing token) are
    dropped so every span is non-empty.

    Returns:
        (response_ids, step_spans) where spans are absolute positions in
        prompt_ids + response_ids.
    """
    step_texts = split_into_step_texts(response)
    offset = len(prompt_ids)

    spans, response_ids, previous_length = [], [], 0
    prefix = ""
    for text in step_texts:
        prefix += text
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if len(prefix_ids) > previous_length:
            spans.append((offset + previous_length, offset + len(prefix_ids)))
            previous_length = len(prefix_ids)
        response_ids = prefix_ids

    return response_ids, spans
