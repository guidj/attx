import tensorflow.compat.v1 as tf

from attx import data
import numpy as np


def create_data_generator(max_length: int):
    assert max_length >= 3

    def generator():
        while True:
            seq = data.create_sequence(np.random.randint(3, max_length + 1))
            yield data.create_example(seq)

    return generator


def create_dataset(max_length: int):
    return tf.data.Dataset.from_generator(
        create_data_generator(max_length),
        output_types=(tf.string, tf.string),
        output_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
    )
