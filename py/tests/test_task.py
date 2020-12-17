from typing import List
import random

import hypothesis
import hypothesis.strategies as st
import pytest
import tensorflow.compat.v1 as tf

from attx import task


@hypothesis.given(st.integers(3, 100))
def test_data_generator_yields_valid_examples(max_length: int):
    gen = task.create_data_generator(max_length, "_")()

    for _ in range(100):
        output_seq, output_label = next(gen)
        assert 0 <= len(output_seq) <= max_length + 1
        assert len(output_label) == 1


@hypothesis.settings(deadline=None, max_examples=10)
@hypothesis.given(st.integers(3, 100))
def test_dataset_yields_valid_examples(session: tf.Session, num_chars: int):
    dataset = task.create_dataset(num_chars, "_", 1)

    iter_op = tf.data.make_one_shot_iterator(dataset).get_next()

    for _ in range(100):
        output_seq, output_label = session.run(iter_op)
        assert 0 <= len(output_seq) <= num_chars + 1
        assert len(output_label) == 1


@hypothesis.given(st.lists(st.characters()), st.integers(0, 100))
def test_pad_sequence_pads_to_max_length(chars: List[str], pad_size: int):
    padded_seq_size = len(chars) + pad_size
    output = task.pad_sequence(chars, max_length=padded_seq_size, token="0xF")

    assert len(output) == padded_seq_size
    assert output[: len(chars)] == chars
    assert output[len(chars) :] == ["0xF"] * pad_size


@hypothesis.given(st.lists(st.characters()))
def test_pad_sequence_does_not_add_to_sequences_longer_than_max_length(
    chars: List[str],
):
    padded_seq_size = len(chars) - random.randint(0, len(chars))
    output = task.pad_sequence(chars, max_length=padded_seq_size, token="0xF")

    assert output == chars


@pytest.fixture
def session() -> tf.Session:
    with tf.Session() as sess:
        yield sess
