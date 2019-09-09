import tensorflow as tf

from debias.utils import ops
from debias.utils.configured import Configured


def compute_nll(logits, labels, mask=None):
  """Computes the NLL of selecting the elements in `labels`

  :param logits: [batch, time, n]
  :param labels: [batch, time, n]
  :param mask: [batch] sequence lengths or [batch, time] binary mask
  :return: [batch, n], the negative log probabilities
  """
  norms = tf.reduce_logsumexp(ops.mask_logits(logits, mask), 1)
  answer_scores = tf.reduce_logsumexp(ops.mask_logits(logits, labels), 1)
  return norms - answer_scores


class QaDebiasLossFunction(Configured):
  """QA debiasing loss functions.

  Built for models that return a start/end logit score for each token
  """

  def compute_qa_loss(self, question_hidden, passage_hidden, logits, bias, labels, mask):
    """
    :param question_hidden: [batch, seq_len, n_q_features] hidden features for question words
    :param passage_hidden: [batch, seq_len, n_p_features] hidden features for passage words
    :param logits: [batch, seq_len, 2] logit start/end score for each word
    :param bias: [batch, seq_len, 2] bias log-probability for the start/end tokens
    :param labels: [batch, seq_len, 2] binary mask of correct start/end tokens
    :param mask: [batch] sequence lengths
    :return: scalar loss
    """
    raise NotImplementedError()


class Plain(QaDebiasLossFunction):
  def compute_qa_loss(self, question_hidden, passage_hidden, logits, bias, labels, mask):
    return tf.reduce_mean(compute_nll(logits, labels, mask))


class BiasProduct(QaDebiasLossFunction):
  def compute_qa_loss(self, question_hidden, passage_hidden, logits, bias, labels, mask):
    logits = tf.nn.log_softmax(logits, 1)
    return tf.reduce_mean(compute_nll(logits+bias, labels, mask))


class Reweight(QaDebiasLossFunction):
  def compute_qa_loss(self, question_hidden, passage_hidden, logits, bias, labels, mask):
    losses = compute_nll(logits+bias, labels, mask)
    weights = tf.reduce_sum(tf.exp(bias) * tf.cast(labels, tf.float32), 1)
    return tf.reduce_sum(losses*weights) / tf.reduce_sum(weights)


class LearnedMixin(QaDebiasLossFunction):
  def __init__(self, w, dim=50):
    self.w = w
    self.dim = dim

  def compute_qa_loss(self, question_hidden, passage_hidden, logits, bias, labels, mask):
    logits = tf.nn.log_softmax(logits, 1)

    p1 = ops.max_pool(ops.affine(question_hidden, self.dim, "q-w", "q-b"), mask)
    p2 = ops.max_pool(ops.affine(passage_hidden, self.dim, "p-w", "p-b"), mask)
    hidden = tf.concat([p1, p2], 1)  # [batch, dim*2]
    factor = ops.affine(hidden, 1, "scale-w", "scale-b")  # [batch, 1]
    factor = tf.nn.softplus(factor)
    bias = bias * tf.expand_dims(factor, 2)

    loss = tf.reduce_mean(compute_nll(bias + logits, labels, mask))

    if self.w == 0:
      return loss

    bias_lp = tf.nn.log_softmax(ops.mask_logits(bias, mask), 1)
    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(bias_lp) * bias_lp, 1))

    return loss + self.w * entropy
