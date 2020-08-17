from typing import Tuple

import tensorflow.compat.v1 as tf


def attention(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Computes attention for 3-D tensors: (batch, sequence_len, dk)
    Where dk is the dimension of each value in the sequence.
    :param query: query tensors, dimensions (batch, sequence_len, dk)
    :param key: key tensors, dimensions (batch, sequence_len, dk)
    :param value: value tensors dimensions (batch, sequence_len, dv)
    :return: attention tensors (batch, sequence_len, sequence_len) and output tensors (batch, sequence_len, dv)
    """
    dk = tf.cast(tf.shape(key)[-1], query.dtype)
    z = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])) / tf.sqrt(dk)
    attention_values = tf.math.softmax(z)
    output = tf.matmul(attention_values, value)
    return attention_values, output
