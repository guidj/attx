from typing import Tuple

import tensorflow.compat.v1 as tf


def attention(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    dk = tf.cast(tf.shape(key)[-1], query.dtype)

    attention_values = tf.math.softmax((query * tf.transpose(key)) / tf.sqrt(dk))
    return attention_values, attention_values * value
