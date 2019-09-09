from typing import Iterable

import tensorflow as tf

from debias.utils import ops
from debias.utils.configured import Configured
from debias.utils.ops import get_shape_tuple


def activation_fn(x, fn_name):
  if fn_name == "relu":
    return tf.nn.relu(x)
  elif fn_name == "tanh":
    return tf.nn.tanh(x)
  elif fn_name is None:
    return x
  else:
    raise NotImplementedError(fn_name)


def _wrap_init(init_fn):
  def wrapped(shape, dtype=None, partition_info=None):
    if partition_info is not None:
      raise ValueError()
    return init_fn(shape, dtype)

  return wrapped


class SequenceMapper(Configured):
    """ (batch, time, in_dim) -> (batch, time, out_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class Mapper(SequenceMapper):
    """ (dim1, dim2, ...., input_dim) -> (dim1, dim2, ...., output_dim) """
    def apply(self, is_train, x, mask=None):
        raise NotImplementedError()


class PoolingLayer(Configured):
  def apply(self, is_train, x, mask):
    raise NotImplementedError()


class MapperSeq(Mapper):
  def __init__(self, layers: Iterable[SequenceMapper]):
    self.layers = list(layers)

  def apply(self, is_train, x, mask=None):
    for i, layer in enumerate(self.layers):
      with tf.variable_scope("layer-%d" % i):
        x = layer.apply(is_train, x, mask)
    return x


def mseq(*layers: Mapper):
  return MapperSeq(layers)


class SequenceMapperSeq(SequenceMapper):
  def __init__(self, layers: Iterable[SequenceMapper]):
    self.layers = list(layers)

  def apply(self, is_train, x, mask=None):
    for i, layer in enumerate(self.layers):
      with tf.variable_scope("layer-%d" % i):
        x = layer.apply(is_train, x, mask)
    return x


def seq(*layers: SequenceMapper):
  return SequenceMapperSeq(layers)


class FullyConnected(Mapper):
  def __init__(self, n_out=None, activation="relu"):
    self.n_out = n_out
    self.activation = activation

  def apply(self, is_train, x, mask=None):
    n_out = self.n_out
    if n_out is None:
      n_out = x.shape.as_list()[-1]
    if self.activation == "glu":
      gate, lin = tf.split(ops.affine(x, n_out*2, "w"), 2, -1)
      gate += tf.get_variable("b", n_out, initializer=tf.zeros_initializer())
      return tf.nn.sigmoid(gate) * lin
    else:
      return activation_fn(ops.affine(x, n_out, "w", bias_name="b"), self.activation)


class VariationalDropout(SequenceMapper):
  def __init__(self, dropout_rate):
    self.dropout_rate = dropout_rate

  def apply(self, is_train, x, mask=None):
    if is_train:
      shape = get_shape_tuple(x)
      return tf.nn.dropout(x, rate=self.dropout_rate, noise_shape=[shape[0], 1, shape[2]])
    else:
      return x


class Dropout(Mapper):
  def __init__(self, dropout_rate):
    self.dropout_rate = dropout_rate

  def apply(self, is_train, x, mask=None):
    if is_train:
      return tf.nn.dropout(x, rate=self.dropout_rate)
    else:
      return x


class MaxPooler(PoolingLayer):
  def apply(self, is_train, x, mask=None):
    return ops.max_pool(x, mask)


class Conv1d(Mapper):
  def __init__(self, num_filters, filter_size, activation="relu", same=False,
               leftpad=None):
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.activation = activation
    self.same = same
    self.leftpad = leftpad

  def apply(self, is_train, x, mask=None):
    x_shape = get_shape_tuple(x)
    dim = x_shape[-1]
    time = x_shape[-2]

    if mask is not None:
      x *= tf.expand_dims(tf.sequence_mask(mask, x_shape[1], tf.float32), 2)

    if self.leftpad:
      if self.same:
        raise ValueError()
      x = tf.pad(x, [[0, 0], [self.filter_size-1, 0], [0, 0]])

    n_filters = self.num_filters
    if self.activation == "glu":
      n_filters *= 2

    if len(x_shape) != 3:
      x = tf.reshape(x, [-1, time, dim])

    filter_ = tf.get_variable("conv1d/filters", shape=[self.filter_size, dim, n_filters],
                              dtype='float')
    out = tf.nn.conv1d(x, filter_, 1, "SAME" if self.same else "VALID")

    if self.activation is not None:
      bias = tf.get_variable("conv1d/bias", shape=[self.num_filters], dtype='float',
                             initializer=tf.zeros_initializer())

      if self.activation == "glu":
        gates, lin = tf.split(out, 2, -1)
        out = tf.nn.sigmoid(gates + bias) * lin
      else:
        out = activation_fn(out + bias, self.activation)

    if len(x_shape) != 3:
      out = tf.reshape(out, x_shape[:-2] + get_shape_tuple(out)[-2:])
    return out


class HighwayLayer(Mapper):
  def __init__(self, layer, transform="tanh"):
    self.layer = layer
    self.transform = transform

  def apply(self, is_train, x, mask=None):
    with tf.variable_scope("layer"):
      out = self.layer.apply(is_train, x, mask)
    dim = out.shape.as_list()[-1]

    if isinstance(self.transform, Mapper) or isinstance(self.transform, SequenceMapper):
      with tf.variable_scope("transform"):
        transform = self.transform.apply(is_train, x, mask)
      gate = ops.affine(x, dim, "w", "b")
    else:
      proj = ops.affine(x, dim * 2, "w", bias_name="b")
      gate, transform = tf.split(proj, 2, 2)
      transform = activation_fn(transform, self.transform)

    gate = tf.sigmoid(gate)
    return transform * (1 - gate) + gate * out
