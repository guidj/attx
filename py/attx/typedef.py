from typing import Union, Dict
import tensorflow as tf

Feature = Union[tf.io.FixedLenFeature, tf.io.VarLenFeature]
FeatureSpec = Dict[str, Feature]
Tensor = Union[tf.Tensor, tf.SparseTensor]
TensorSpec = Dict[str, Tensor]
