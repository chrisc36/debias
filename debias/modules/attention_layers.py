from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor

from debias.utils import ops
from debias.utils.configured import Configured


def compute_attention_mask(x_mask, mem_mask, x_word_dim, key_word_dim):
    """ computes a (batch, x_word_dim, key_word_dim) bool mask for clients that want masking """
    if x_mask is None and mem_mask is None:
        return None
    elif x_mask is None:
        return tf.expand_dims(tf.sequence_mask(mem_mask, key_word_dim), 1)
    elif mem_mask is None:
        return tf.expand_dims(tf.sequence_mask(x_mask, x_word_dim), 2)

    x_mask = tf.sequence_mask(x_mask, x_word_dim)
    mem_mask = tf.sequence_mask(mem_mask, key_word_dim)
    join_mask = tf.logical_and(tf.expand_dims(x_mask, 2), tf.expand_dims(mem_mask, 1))
    return join_mask


class SimilarityLayer(Configured):
    """
    Computes a pairwise score between elements in each sequence
    (batch, time1, dim1], (batch, time2, dim2) -> (batch, time1, time2)
    """
    def get_scores(self, is_train, tensor_1, tensor_2):
        raise NotImplementedError

    def get_logit_masked_scores(self, is_train, tensor_1, tensor_2, mask1, mask2):
      atten = self.get_scores(is_train, tensor_1, tensor_2)
      dim1, dim2 = ops.get_shape_tuple(atten)[1:]
      mask = compute_attention_mask(mask1, mask2, dim1, dim2)
      return ops.mask_logits(atten, mask)


class WeightedDot(SimilarityLayer):
  """ Function used by BiDaF, bi-linear with an extra component for the dots of the vectors """
  def get_scores(self, is_train, x, keys):
    dim = keys.shape.as_list()[-1]
    key_w = tf.get_variable("key_w", shape=dim, dtype=tf.float32)
    key_logits = tf.tensordot(keys, key_w, axes=[[2], [0]])  # (batch, key_len)

    x_w = tf.get_variable("input_w", shape=dim, dtype=tf.float32)
    x_logits = tf.tensordot(x, x_w, axes=[[2], [0]])  # (batch, x_len)

    dot_w = tf.get_variable("dot_w", shape=dim, dtype=tf.float32)

    # Compute x * dot_weights first, the batch mult with x
    x_dots = x * tf.expand_dims(tf.expand_dims(dot_w, 0), 0)
    dot_logits = tf.matmul(x_dots, keys, transpose_b=True)

    return dot_logits + tf.expand_dims(key_logits, 1) + tf.expand_dims(x_logits, 2)

  def __setstate__(self, state):
    if "dropout" in state:  # FIXME
      del state["dropout"]
      del state["vdropout"]
    super().__setstate__(state)


class AttentionBiFuse(Configured):
  def __init__(self, similarity: SimilarityLayer):
    self.similarity = similarity

  def apply(self, is_train, seq1, seq2, mask1, mask2) -> Tuple[Tensor, Tensor]:
    with tf.variable_scope("sim"):
      atten = self.similarity.get_logit_masked_scores(is_train, seq1, seq2, mask1, mask2)

    atten1 = tf.matmul(tf.nn.softmax(atten), seq2)
    atten2 = tf.matmul(tf.nn.softmax(tf.transpose(atten, [0, 2, 1])), seq1)

    s1 = tf.concat([seq1, atten1, seq1 * atten1], 2)
    s2 = tf.concat([seq2, atten2, seq2 * atten2], 2)
    return s1, s2


class BiAttention(Configured):
  """ Bi-Directonal Attention from https://arxiv.org/abs/1611.01603 """

  def __init__(self, sim: SimilarityLayer, q2c: bool=True, query_dots: bool=True):
    self.sim = sim
    self.q2c = q2c
    self.query_dots = query_dots

  def apply(self, is_train, src, other, src_mask=None, other_mask=None):
    dist_matrix = self.sim.get_logit_masked_scores(is_train, src, other, src_mask, other_mask)
    query_probs = tf.nn.softmax(dist_matrix)  # probability of each mem_word per x_word

    # Batch matrix multiplication to get the attended vectors
    select_query = tf.matmul(query_probs, other)  # (batch, x_words, q_dim)

    if not self.q2c:
      if self.query_dots:
        return tf.concat([src, select_query, src * select_query], axis=2)
      else:
        return tf.concat([src, select_query], axis=2)

    # select query-to-context
    context_dist = tf.reduce_max(dist_matrix, axis=2)  # (batch, x_word``s)
    context_probs = tf.nn.softmax(context_dist)  # (batch, x_words)
    select_context = tf.einsum("ai,aik->ak", context_probs, src)  # (batch, x_dim)
    select_context = tf.expand_dims(select_context, 1)

    if self.query_dots:
      return tf.concat([src, select_query, src * select_query, src * select_context], axis=2)
    else:
      return tf.concat([src, select_query, src * select_context], axis=2)
