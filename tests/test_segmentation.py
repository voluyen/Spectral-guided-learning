"""Segmentation tests use a whitespace-free character tokenizer so token offsets are
exactly predictable; the real-tokenizer round-trip is checked by data_prep.py itself."""

from segmentation import segment_response_token_spans, split_into_step_texts


class CharTokenizer:
    """Minimal stand-in: one token per character."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def test_split_preserves_original_text():
    response = "Step one. Step two.\n\nStep three."
    assert "".join(split_into_step_texts(response)) == response


def test_split_on_sentence_boundaries():
    assert split_into_step_texts("Step one. Step two. Step three.") == [
        "Step one. ",
        "Step two. ",
        "Step three.",
    ]


def test_split_on_newline_after_punctuation():
    assert len(split_into_step_texts("First part?\nSecond part!\nThird part.")) == 3


def test_split_after_closing_brace_or_bracket():
    assert len(split_into_step_texts(r"So \boxed{7} Therefore we stop.")) == 2
    assert len(split_into_step_texts(r"Hence \] Now continue.")) == 2


def test_no_split_without_capital_letter():
    assert len(split_into_step_texts("value is 3. then we continue")) == 1


def test_no_split_without_whitespace():
    assert len(split_into_step_texts("3.5 is the answer")) == 1


def test_empty_response_yields_no_steps():
    assert split_into_step_texts("") == []


def test_spans_are_contiguous_and_offset_by_prompt():
    tokenizer = CharTokenizer()
    prompt_ids = [1, 2, 3]
    response = "Aaaaaaaaaa. Bbbbbbbbbb. Cccccccccc."

    response_ids, spans = segment_response_token_spans(tokenizer, prompt_ids, response)

    assert len(spans) == 3
    assert spans[0][0] == len(prompt_ids)
    assert spans[-1][1] == len(prompt_ids) + len(response_ids)
    for previous, following in zip(spans, spans[1:]):
        assert previous[1] == following[0]  # no gaps, no overlaps


def test_short_steps_are_kept_not_merged():
    tokenizer = CharTokenizer()
    response = "Aaaaaaaaaaaa. B. Cccccccccccc."

    _, spans = segment_response_token_spans(tokenizer, [], response)

    assert len(spans) == 3  # short middle step survives — no merging
    assert all(end > start for start, end in spans)


def test_unsplittable_response_yields_single_span():
    tokenizer = CharTokenizer()
    _, spans = segment_response_token_spans(tokenizer, [], "one long line without boundaries")
    assert len(spans) == 1
