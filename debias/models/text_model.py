from typing import Tuple

import tensorflow as tf

from debias.datasets.training_data_loader import HYPOTHESIS_KEY, PREMISE_KEY, HYPOTHESIS_LEN_KEY, \
  PREMISE_LEN_KEY
from debias.modules.word_and_char_encoder import WordAndCharEncoder
from debias.utils import configured
from debias.utils.tokenizer import Tokenizer


class EncodedText:
  """Text encoded as vectors."""

  def __init__(self, embeddings, mask):
    self.embeddings = embeddings
    self.mask = mask


class TextModel(configured.Configured):
  """Model that takes a pair of text inputs.

  This class takes care of "tensorizing" text into embeddings, subclasses decide
  what to do with those embeddings
  """

  def __init__(self, tokenizer: Tokenizer, text_encoder: WordAndCharEncoder):
    self.tokenizer = tokenizer
    self.text_encoder = text_encoder

  def set_vocab(self, voc):
    if voc is not None:
      voc = list(voc)
    self.text_encoder.set_vocab(voc)

  def tensorize_fn(self):
    """Build a function to pre-process tokenized data with that can be used with tf.map"""

    fn = self.text_encoder.tensorize_fn()

    def map_fn(x):
      for key in [PREMISE_KEY, HYPOTHESIS_KEY]:
        t = x[key]
        x[key + "_tensors"] = fn(t)
        x[key + "_len"] = tf.shape(t)[-1]
      return x

    return map_fn

  def get_tokenizer(self):
    """Returns a tokenizer to use on raw text."""
    return self.tokenizer

  def apply(self, is_train, features, labels):
    """Returns a tensor containing the model's output.

    Also should add the loss tf.GraphKeys.LOSSES if `is_train` is True

    Args:
      is_train: train or evaluation mode
      features: batched feature dictionary as built by `self.tensorize`
      labels: tensor or nested set of tensors that are the example labels

    Returns:
      the models tensor output
    """
    raise NotImplementedError()

  def get_text_embeddings(self, is_train, features) -> Tuple[EncodedText, EncodedText]:
    """Build text embeddings from `features`, where `features` is the batch, dictionary
    of tensors built by `self.tensorize_fn`"""

    p_tensors = features[PREMISE_KEY + "_tensors"]
    h_tensors = features[HYPOTHESIS_KEY + "_tensors"]

    with tf.variable_scope("embed"):
      with tf.variable_scope("word-embed"):
        p_wembed, h_wembed = self.text_encoder.embed_words(
          is_train, [p_tensors, h_tensors])

    h_len = features[HYPOTHESIS_LEN_KEY]
    h_embed = EncodedText(h_wembed, h_len)

    p_len = features[PREMISE_LEN_KEY]
    p_embed = EncodedText(p_wembed, p_len)

    return h_embed, p_embed
