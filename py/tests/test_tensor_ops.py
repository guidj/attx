import pytest
import tensorflow as tf
import numpy as np

from attx import tensor_ops


def test_single_example_batch_single_step_sequence_with_high_dimension():
    # (?, k, dk) = (1, 1, 4)
    query_1 = [[1, 2, 3, 4]]
    key_1 = [[1, 1, 1, 1]]
    value_1 = [[10, 10, 10, 10]]

    query = tf.cast([query_1], tf.float32)
    key = tf.cast([key_1], tf.float32)
    value = tf.cast([value_1], tf.float32)

    expected_att_1 = [[1.0]]
    expected_output_1 = [[10.0, 10.0, 10.0, 10.0]]

    expected_attention = np.array([expected_att_1])
    expected_value = np.array([expected_output_1])

    output_attention, output_value = tensor_ops.attention(query, key, value)

    np.testing.assert_array_almost_equal(
        output_attention, expected_attention, decimal=3,
    )
    np.testing.assert_array_almost_equal(output_value, expected_value, decimal=3)


def test_single_example_batch_multi_step_sequence_with_high_dimension():
    # (?, k, dk) = (1, 2, 4)
    query_1 = [[1, 3, 5, 7], [2, 4, 6, 8]]
    key_1 = [[1, 1, 1, 1], [1, 1, 1, 1]]
    value_1 = [[10, 10, 10, 10], [50, 50, 50, 50]]

    query = tf.cast([query_1], tf.float32)
    key = tf.cast([key_1], tf.float32)
    value = tf.cast([value_1], tf.float32)

    expected_att_1 = [[0.5, 0.5], [0.5, 0.5]]
    expected_output_1 = [[30.0, 30.0, 30.0, 30.0], [30.0, 30.0, 30.0, 30.0]]

    expected_attention = np.array([expected_att_1])
    expected_value = np.array([expected_output_1])

    output_attention, output_value = tensor_ops.attention(query, key, value)

    np.testing.assert_array_almost_equal(
        output_attention, expected_attention, decimal=3,
    )
    np.testing.assert_array_almost_equal(output_value, expected_value, decimal=3)


def test_single_example_batch_multi_step_sequence_with_single_dimension():
    # (?, k, dk) = (1, 4, 1)
    query_1 = [[1], [2], [3], [4]]
    key_1 = [[1], [1], [1], [1]]
    value_1 = [10], [10], [10], [10]

    query = tf.cast([query_1], tf.float32)
    key = tf.cast([key_1], tf.float32)
    value = tf.cast([value_1], tf.float32)

    expected_att_1 = [
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
        [1 / 4, 1 / 4, 1 / 4, 1 / 4],
    ]
    expected_output_1 = [[10], [10], [10], [10]]

    expected_attention = np.array([expected_att_1])
    expected_value = np.array([expected_output_1])

    output_attention, output_value = tensor_ops.attention(query, key, value)

    np.testing.assert_array_almost_equal(
        output_attention, expected_attention, decimal=3,
    )
    np.testing.assert_array_almost_equal(output_value, expected_value, decimal=3)


def test_multi_example_batch_multi_step_sequence_with_high_dimension():
    # (?, k, dk) = (2, 2, 4)
    query_1 = [[1, 3, 5, 7], [2, 4, 6, 8]]
    query_2 = [[1, 3, 5, 7], [2, 4, 6, 8]]
    key_1 = [[1, 1, 1, 1], [1, 1, 1, 1]]
    key_2 = [[1, 2, 1, 2], [2, 1, 2, 1]]
    value_1 = [[10, 10, 10, 10], [50, 50, 50, 50]]
    value_2 = [[10, 10, 10, 10], [50, 50, 50, 50]]

    query = tf.cast([query_1, query_2], tf.float32)
    key = tf.cast([key_1, key_2], tf.float32)
    value = tf.cast([value_1, value_2], tf.float32,)

    expected_att_1 = [[0.5, 0.5], [0.5, 0.5]]
    expected_att_2 = [[0.881, 0.119], [0.881, 0.119]]
    expected_output_1 = [[30.0, 30.0, 30.0, 30.0], [30.0, 30.0, 30.0, 30.0]]
    expected_output_2 = [
        [369 / 25, 369 / 25, 369 / 25, 369 / 25],
        [369 / 25, 369 / 25, 369 / 25, 369 / 25],
    ]

    expected_attention = np.array([expected_att_1, expected_att_2])
    expected_value = np.array([expected_output_1, expected_output_2])

    output_attention, output_value = tensor_ops.attention(query, key, value)

    np.testing.assert_array_almost_equal(
        output_attention, expected_attention, decimal=3,
    )
    np.testing.assert_array_almost_equal(output_value, expected_value, decimal=2)


def test_single_example_batch_multi_step_sequence_with_high_dimension_and_different_value_dimension():
    # (?, k, dk) = (1, 2, 4)
    query_1 = [[1, 3, 5, 7], [2, 4, 6, 8]]
    key_1 = [[1, 1, 1, 1], [1, 1, 1, 1]]
    # (?, k, dv) = (1, 2, 5)
    value_1 = [[10, 10, 10, 10, 10], [50, 50, 50, 50, 50]]

    query = tf.cast([query_1], tf.float32)
    key = tf.cast([key_1], tf.float32)
    value = tf.cast([value_1], tf.float32)

    expected_att_1 = [[0.5, 0.5], [0.5, 0.5]]
    expected_output_1 = [[30.0, 30.0, 30.0, 30.0, 30.0], [30.0, 30.0, 30.0, 30.0, 30.0]]

    expected_attention = np.array([expected_att_1])
    expected_value = np.array([expected_output_1])

    output_attention, output_value = tensor_ops.attention(query, key, value)

    np.testing.assert_array_almost_equal(
        output_attention, expected_attention, decimal=3,
    )
    np.testing.assert_array_almost_equal(output_value, expected_value, decimal=3)
