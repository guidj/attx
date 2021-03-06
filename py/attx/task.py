import os.path
import tempfile
import uuid
from typing import Optional, Callable, Tuple, Dict, Any, List, Generator, List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from attx import data
from attx.typedef import FeatureSpec, Tensor
from attx.typedef import TensorSpec


SEQ_LEN = 50
EMBEDDING_SIZE = 8
MIN_SEQ_LEN = 3


def create_train_and_eval_input_functions(
    num_chars: int, batch_size: int
) -> Tuple[Callable[[], tf.data.Dataset], Callable[[], tf.data.Dataset]]:
    """
    Generates zero-param input functions for training and evaluation.
    """
    train_input_fn = create_input_fn(num_chars, batch_size)
    eval_input_fn = create_input_fn(num_chars, 32)

    return train_input_fn, eval_input_fn


def create_input_fn(num_chars: int, batch_size: int) -> Callable[[], tf.data.Dataset]:
    """
    Returns a zero-param function that returns a dataset of examples.
    :param num_chars: num alphabet chars in sequence. Output seq is sized num_chars + 1.
    :param batch_size: tf.Dataset will returns batches of examples of specified size.
    """

    def input_fn() -> tf.data.Dataset:
        return create_dataset(num_chars, data.NONE_TOKEN, batch_size)

    return input_fn


def create_dataset(
    num_chars: int, padding_token: str, batch_size: int
) -> tf.data.Dataset:
    """
    Returns a tf.Dataset of input, label dictionaries of format:
        {"sequence": List[str]}, {"target": str}
    :param num_chars:num alphabet chars in sequence. Output seq is sized num_chars + 1.
    :param padding_token: value to pad every generated sequence that is smaller than max_length.
    :param batch_size: tf.Dataset will returns batches of examples of specified size.
    """

    def create_example_dictionaries(
        x: Tensor, y: Tensor
    ) -> Tuple[TensorSpec, tf.Tensor]:
        return {"sequence": x}, y

    dataset = tf.data.Dataset.from_generator(
        create_data_generator(num_chars, padding_token),
        output_types=(tf.string, tf.string),
        output_shapes=(tf.TensorShape([num_chars + 1]), tf.TensorShape([])),
    )

    return dataset.map(create_example_dictionaries).batch(batch_size)


def create_data_generator(
    num_chars: int, padding_token: str
) -> Generator[Tuple[List[str], str], None, None]:
    """
    Returns a zero-param function that can produces a generator of data.
    This generator creates a sequence, and pads it to max_length.
    It retuns (padded_seq, label) tuples.
    Padded seq will be of size max_length + 1 (added next token).
    :param num_chars:num alphabet chars in sequence. Output seq is sized num_chars + 1.
    :param padding_token: value to pad every generated sequence that is smaller than max_length.
    """
    assert num_chars >= MIN_SEQ_LEN
    max_length = num_chars + 1

    def generator():
        while True:
            seq = data.create_sequence(
                np.random.randint(MIN_SEQ_LEN, num_chars)
                if num_chars > MIN_SEQ_LEN
                else MIN_SEQ_LEN
            )
            new_seq, label = data.create_example(seq)
            yield pad_sequence(new_seq, max_length, padding_token), label

    return generator


def pad_sequence(x: List[str], max_length: int, token: str) -> List[str]:
    return x + ([token] * (max_length - len(x)))


def features_and_label_spec() -> Tuple[FeatureSpec, FeatureSpec]:
    features = {"sequence": tf.io.FixedLenFeature([SEQ_LEN], dtype=tf.string)}
    labels = {"target": tf.io.FixedLenFeature([], dtype=tf.string)}
    return features, labels


def create_estimator(
    model_dir: str, config: Optional[tf.estimator.RunConfig], params: Dict[str, Any]
) -> tf.estimator.Estimator:
    """
    Returns an estimator.
    """
    layers_config = params["layers_config"]
    embedding_size = params["embedding_size"]

    token_id = tf.feature_column.categorical_column_with_vocabulary_list(
        key="sequence", vocabulary_list=data.VOCAB, num_oov_buckets=0
    )
    sequence_feature = tf.feature_column.embedding_column(
        token_id, dimension=embedding_size, combiner="mean"
    )

    columns = [sequence_feature]

    estimator = tf.estimator.DNNClassifier(
        feature_columns=columns,
        hidden_units=layers_config,
        n_classes=data.VOCAB_SIZE,
        optimizer=tf.keras.optimizers.Adam(),
        label_vocabulary=data.VOCAB,
        activation_fn=tf.nn.elu,
        dropout=0.2,
        batch_norm=True,
    )
    return estimator


def create_hooks(log_dir: str) -> List[tf.estimator.SessionRunHook]:
    step_counter_hook = tf.estimator.StepCounterHook(every_n_steps=100)
    return [step_counter_hook]


def train():
    # params
    model_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    params = {"layers_config": [8, 8, 8], "embedding_size": 16}
    config = tf.estimator.RunConfig()

    # setup
    estimator = create_estimator(model_dir, config, params)
    train_input_fn, eval_input_fn = create_train_and_eval_input_functions(SEQ_LEN, 64)

    hooks = create_hooks(model_dir)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, hooks=hooks, max_steps=1000
    )
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    train()
