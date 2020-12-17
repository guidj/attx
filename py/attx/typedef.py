from typing import Union, Dict
import tensorflow.compat.v1 as tf

Feature = Union[tf.FixedLenFeature, tf.VarLenFeature]
FeatureSpec = Dict[str, Feature]
Tensor = Union[tf.Tensor, tf.SparseTensor]
TensorSpec = Dict[str, Tensor]
