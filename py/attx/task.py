import os.path
import tempfile
import uuid
from typing import Optional, Callable, Tuple, Dict, Any, List, Generator, List

import numpy as np
import tensorflow.compat.v1 as tf
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
    ) -> Tuple[TensorSpec, TensorSpec]:
        return {"sequence": x}, {"target": y}

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
    features = {"sequence": tf.FixedLenFeature([SEQ_LEN], dtype=tf.string)}
    labels = {"target": tf.FixedLenFeature([], dtype=tf.string)}
    return features, labels


def model_fn(
    features: TensorSpec,
    labels: TensorSpec,
    mode: Optional[tf.estimator.ModeKeys],
    params: Dict[str, Any],
) -> tf.estimator.EstimatorSpec:
    layers_config = params["layers_config"]
    embedding_size = params["embedding_size"]
    embedding_table = create_token_lookup(data.VOCAB_INDEX)
    alphabet_embeddings = create_embeddings(embedding_size)

    token_ids = embedding_table.lookup(features["sequence"], name="example/sequence/tokens")
    target_ids = embedding_table.lookup(labels["target"], name="example/target/tokens")
    # examples x rows x dim-size => examples x dim-size
    embeddings = tf.nn.embedding_lookup(alphabet_embeddings, token_ids, name="example/sequence/embeddings")
    # average embeddings per example
    embeddings_agg = tf.reduce_mean(embeddings, axis=1, name="example/sequence/agg_embeddings")
    targets = tf.one_hot(target_ids, depth=data.VOCAB_SIZE, name="example/target/OHE")

    layers = []
    prev_layer = embeddings_agg
    for idx, size in enumerate(layers_config):
        layer = tf.layers.dense(prev_layer, units=size, name=f"fcc/{idx}")
        prev_layer = tf.nn.elu(layer, name=f"fcc/{idx}/elu")
        layers.append(prev_layer)

    logits = tf.layers.dense(layers[-1], units=data.VOCAB_SIZE, name="classifier/logits")
    predictions = tf.nn.softmax(logits, name="classifier/softmax")

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=logits, name="loss")
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    metrics_ops = {
        "precision": tf.metrics.precision(targets, predictions),
        "recall": tf.metrics.recall(targets, predictions),
    }

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics_ops,
        export_outputs={
            "labels": tf.estimator.export.ClassificationOutput(
                scores=predictions, classes=tf.constant(data.VOCAB)
            )
        },
    )


def create_embeddings(embedding_size: int) -> tf.Variable:
    alphabet_embeddings = tf.get_variable(
        "alphabet-embeddings", [data.VOCAB_SIZE, embedding_size]
    )
    return alphabet_embeddings


def create_token_lookup(
    vocabulary_index: Dict[str, int], num_oov_buckets: int = 1
) -> lookup_ops.LookupInterface:

    keys, values = zip(*vocabulary_index.items())

    initializer = tf.lookup.KeyValueTensorInitializer(
        tf.constant(keys, dtype=tf.string),
        tf.constant(values, dtype=tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64,
    )

    return tf.lookup.StaticVocabularyTable(initializer, num_oov_buckets)


def create_estimator(
    model_dir: str, config: Optional[tf.estimator.RunConfig], params: Dict[str, Any]
) -> tf.estimator.Estimator:
    return tf.estimator.Estimator(
        model_fn, model_dir=model_dir, config=config, params=params,
    )


def train():
    # params
    model_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    params = {"layers_config": [8, 8, 8], "embedding_size": 16}
    config = tf.estimator.RunConfig()

    # setup
    estimator = create_estimator(model_dir, config, params)
    train_input_fn, eval_input_fn = create_train_and_eval_input_functions(SEQ_LEN, 64)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    train()
    # print(create_model(sample(), [3, 4, 2]))
