import argparse
import dataclasses
import logging
import os.path
import tempfile
import uuid
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from attx import data
from attx.typedef import FeatureSpec, Tensor, TensorSpec

SEQ_LEN = 50
MIN_SEQ_LEN = 3
DEFAULT_STEPS = 500

DNN = "dnn"
RNN = "rnn"
MEAN = "mean"
SQRTN = "sqrtn"
SUM = "sum"
ARCH = [DNN, RNN]
EMBEDDING_REDUCE = [MEAN, SQRTN, SUM]


@dataclasses.dataclass
class ModelArgs:
    embedding_size: int
    embedding_method: str
    arch: str


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser("Next Token: Supervised Learning")
    arg_parser.add_argument("--arch", type=str, default="dnn", choices=ARCH)
    arg_parser.add_argument("--embedding-size", type=int, default=16)

    arg_parser.add_argument(
        "--embedding-method", type=str, default="mean", choices=EMBEDDING_REDUCE
    )
    arg_parser.add_argument("--batch-size", type=int, default=64)
    arg_parser.add_argument("--max-steps", type=int, default=1000000)
    arg_parser.add_argument("--eval-steps", type=int, default=10)
    arg_parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(tempfile.gettempdir(), str(uuid.uuid4())),
    )
    args, _ = arg_parser.parse_known_args()

    logging.info("Parsed arguments:")
    for key, value in vars(args).items():
        logging.info("\t%s: %s", key, value)
    return args


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
    model_dir: str,
    config: Optional[tf.estimator.RunConfig],
    params: Dict[str, Any],
    model_args: ModelArgs,
) -> tf.estimator.Estimator:
    """
    Returns an estimator.
    """
    layers_config = params["layers_config"]

    if model_args.arch == DNN:
        token_id = tf.feature_column.categorical_column_with_vocabulary_list(
            key="sequence", vocabulary_list=data.VOCAB, num_oov_buckets=0
        )
        sequence_feature = tf.feature_column.embedding_column(
            token_id,
            dimension=model_args.embedding_size,
            combiner=model_args.embedding_method,
        )
        columns = [sequence_feature]
        estimator = tf.estimator.DNNClassifier(
            model_dir=model_dir,
            config=config,
            feature_columns=columns,
            hidden_units=layers_config,
            n_classes=data.VOCAB_SIZE,
            optimizer=tf.keras.optimizers.Adam(),
            label_vocabulary=data.VOCAB,
            activation_fn=tf.nn.elu,
            dropout=0.0,
            batch_norm=False,
        )
    elif model_args.arch == RNN:
        token_id = tf.feature_column.sequence_categorical_column_with_vocabulary_list(
            key="sequence", vocabulary_list=data.VOCAB, num_oov_buckets=0
        )
        sequence_feature = tf.feature_column.embedding_column(
            token_id, dimension=model_args.embedding_size
        )
        columns = [sequence_feature]
        estimator = tf.estimator.experimental.RNNClassifier(
            model_dir=model_dir,
            config=config,
            sequence_feature_columns=columns,
            units=layers_config,
            cell_type="lstm",
            n_classes=data.VOCAB_SIZE,
            label_vocabulary=data.VOCAB,
            optimizer="Adagrad",
        )
    else:
        raise ValueError(f"Arch {model_args.arch} isn't recognized")
    return estimator


def create_hooks(log_dir: str) -> List[tf.estimator.SessionRunHook]:
    del log_dir
    step_counter_hook = tf.estimator.StepCounterHook(every_n_steps=DEFAULT_STEPS)
    return [step_counter_hook]


def auc_fn(labels, predictions):
    auc_metric = tf.keras.metrics.AUC(name="auc")

    def equality(label_class_ids: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        label, class_ids = label_class_ids
        return tf.equal(label, class_ids)

    indices = tf.where(
        tf.map_fn(
            equality,
            elems=(labels, predictions["all_classes"]),
            fn_output_signature=tf.bool,
        )
    )
    y_true = tf.one_hot(indices[:, 1], depth=data.VOCAB_SIZE)
    y_true = tf.squeeze(y_true)
    auc_metric.update_state(y_true=y_true, y_pred=predictions["probabilities"])
    return {"auc": auc_metric}


def add_metrics(estimator: tf.estimator.Estimator) -> tf.estimator.Estimator:
    return tf.estimator.add_metrics(estimator, auc_fn)


def train(
    model_dir: str,
    max_steps: int,
    eval_steps: int,
    batch_size: int,
    model_args: ModelArgs,
):
    # params
    params = {"layers_config": [12, 12, 12]}
    config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=DEFAULT_STEPS,
        # eval run when a checkpoint is saved
        save_checkpoints_steps=DEFAULT_STEPS * 10,
        log_step_count_steps=DEFAULT_STEPS,
    )

    # setup
    estimator = create_estimator(model_dir, config, params, model_args)
    estimator = add_metrics(estimator)
    train_input_fn, eval_input_fn = create_train_and_eval_input_functions(
        SEQ_LEN, batch_size=batch_size
    )

    hooks = create_hooks(model_dir)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, hooks=hooks, max_steps=max_steps
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=eval_steps,
        start_delay_secs=0,
        throttle_secs=0,
    )

    # run train/eval loop
    tf.estimator.train_and_evaluate(
        estimator, train_spec=train_spec, eval_spec=eval_spec
    )


if __name__ == "__main__":
    args = parse_args()
    train(
        model_dir=args.model_dir,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        model_args=ModelArgs(
            embedding_size=args.embedding_size,
            embedding_method=args.embedding_method,
            arch=args.arch,
        ),
    )
