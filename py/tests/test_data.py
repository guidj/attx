import collections

import hypothesis
import hypothesis.strategies as st

from attx import data


@hypothesis.given(st.integers(3, 100))
def test_sequence_matches_expected_length(length: int):
    assert len(data.create_sequence(length)) == length


@hypothesis.given(st.integers(3, 100))
def test_sequence_contains_uppercase_alphabet_letters_only(length: int):
    output = [
        token
        for token in data.create_sequence(length)
        if not (ord("A") <= ord(token) <= ord("Z"))
    ]
    assert len(output) == 0


@hypothesis.given(st.integers(3, 100))
def test_example_contains_next_token(length: int):
    seq = sequence(length)
    output_seq, _ = data.create_example(seq)
    assert "NTKN" in output_seq


@hypothesis.given(st.integers(3, 100))
def test_example_preserves_seq_elements_positions(length: int):
    seq = [str(token) for token in range(length)]
    output_seq, _ = data.create_example(seq)
    output_seq.remove("NTKN")
    assert output_seq == seq


@hypothesis.given(st.integers(3, 100))
def test_example_label_follows_next_token(length: int):
    seq = [str(token) for token in range(length)]
    output_seq, output_label = data.create_example(seq)
    assert output_label == seq[output_seq.index("NTKN")]
    assert output_seq[output_seq.index("NTKN") + 1] == seq[output_seq.index("NTKN")]


@hypothesis.given(st.integers(3, 100))
def test_example_next_token_is_positioned_between_two_chars(length: int):
    seq = [str(token) for token in range(length)]
    output_seq, _ = data.create_example(seq)
    assert 0 < output_seq.index("NTKN") < length + 1


@hypothesis.given(st.integers(3, 100))
def test_example_contains_no_repeated_next_token(length: int):
    seq = [str(token) for token in range(length)]
    output_seq, _ = data.create_example(seq)
    assert collections.Counter(output_seq)["NTKN"] == 1


def sequence(length: int):
    return [str(token) for token in range(length)]
