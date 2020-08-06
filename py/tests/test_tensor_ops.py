import pytest
import tensorflow.compat.v1 as tf
import numpy as np

from attx import tensor_ops


@pytest.fixture
def session():
    with tf.Session() as sess:
        yield sess


def test_simple_case(session: tf.Session):
    query = tf.constant([1, 3, 6], dtype=tf.float32)
    key = tf.constant([1, 1, 1], dtype=tf.float32)
    value = tf.constant([10, 10, 10], dtype=tf.float32)

    output_op = tensor_ops.attention(query, key, value)

    expected_attention, expected_value = (
        np.array([0.00925714, 0.0523235, 0.93841936]),
        np.array([0.0925714, 0.523235, 9.384194]),
    )
    output_attention, output_value = session.run(output_op)

    np.testing.assert_array_almost_equal(
        output_attention, expected_attention, decimal=6,
    )
    np.testing.assert_array_almost_equal(output_value, expected_value, decimal=6)
