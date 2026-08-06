"""Step segmentation of long CoT responses with exact token-offset boundaries.

The paper specifies "standardized step-wise segmentation" without details; we split at
logical boundaries — after sentence-ending punctuation (or a closing brace/bracket, which
ends inline LaTeX) followed by whitespace and a capital letter. The split is deterministic
and preserves semantically complete steps. No post-hoc merging is applied: every boundary
found by the rule becomes a step, so segmentation depends only on the text itself.

Boundaries are found in character space, then mapped to token indices via the character
offsets the tokenizer reports for its single, canonical pass over the response. The
response is never tokenized piecewise: BPE is context-dependent, so tokenizing steps
separately and concatenating yields token ids the model was never pretrained on (a lone
whitespace token at each step start instead of a merged " Word" token).
"""

import bisect
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


def encode_with_offsets(tokenizer, text: str) -> tuple[list[int], list[int]]:
    """Tokenize `text` once, returning its ids and each token's starting character index.

    Requires a fast (Rust-backed) tokenizer, which every `AutoTokenizer` used here is.
    """
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoding["input_ids"], [start for start, _ in encoding["offset_mapping"]]


def step_token_spans(
    response: str,
    token_starts: list[int],
    offset: int = 0,
) -> list[tuple[int, int]]:
    """Map the response's step boundaries onto token indices, shifted by `offset`.

    A boundary landing inside a token assigns that whole token to the later step, since
    `bisect_left` returns the first token starting at or after the boundary. Zero-token
    steps (two boundaries falling within one token) are dropped so every span is non-empty.
    """
    cuts, position = [], 0
    for text in split_into_step_texts(response):
        cuts.append(position)
        position += len(text)

    bounds = [bisect.bisect_left(token_starts, cut) for cut in cuts] + [len(token_starts)]
    return [
        (offset + start, offset + end)
        for start, end in zip(bounds, bounds[1:])
        if end > start
    ]


def segment_response_token_spans(
    tokenizer,
    prompt_ids: list[int],
    response: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize a response and map each step to an absolute token span.

    Returns:
        (response_ids, step_spans) where spans are absolute positions in
        prompt_ids + response_ids.
    """
    response_ids, token_starts = encode_with_offsets(tokenizer, response)
    return response_ids, step_token_spans(response, token_starts, offset=len(prompt_ids))
