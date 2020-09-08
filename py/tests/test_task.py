import hypothesis
import hypothesis.strategies as st
import pytest

import tensorflow.compat.v1 as tf
from attx import task


@hypothesis.given(st.integers(3, 100))
def test_data_generator_yields_valid_examples(max_length: int):
    gen = task.create_data_generator(max_length)()

    for i in range(100):
        output_seq, output_label = next(gen)
        assert 0 <= len(output_seq) <= max_length + 1
        assert len(output_label) == 1


@hypothesis.settings(deadline=1000, max_examples=10)
@hypothesis.given(st.integers(3, 100))
def test_dataset_yields_valid_examples(session: tf.Session, max_length: int):
    dataset = task.create_dataset(max_length)

    iter_op = tf.data.make_one_shot_iterator(dataset).get_next()

    for i in range(100):
        output_seq, output_label = session.run(iter_op)
        assert 0 <= len(output_seq) <= max_length + 1
        assert len(output_label) == 1


@pytest.fixture
def session() -> tf.Session:
    with tf.Session() as sess:
        yield sess
