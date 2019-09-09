from typing import Optional

import tensorflow as tf

from debias.models.text_model import TextModel
from debias.modules.attention_layers import BiAttention
from debias.modules.layers import SequenceMapper
from debias.modules.qa_debias_loss_functions import QaDebiasLossFunction
from debias.modules.word_and_char_encoder import WordAndCharEncoder
from debias.utils import ops
from debias.utils.tokenizer import Tokenizer


class TextPairQaDebiasingModel(TextModel):
  def __init__(self,
               tokenizer: Tokenizer,
               text_encoder: WordAndCharEncoder,
               map_embed: Optional[SequenceMapper],
               fuse_layer: BiAttention,
               post_process_layer: SequenceMapper,
               debias_loss_fn: QaDebiasLossFunction
               ):
    super().__init__(tokenizer, text_encoder)
    self.map_embed = map_embed
    self.fuse_layer = fuse_layer
    self.post_process_layer = post_process_layer
    self.debias_loss_fn = debias_loss_fn

  def apply(self, is_train, features, labels):
    hypoth, premise = self.get_text_embeddings(is_train, features)
    q_embed, q_mask = hypoth.embeddings, hypoth.mask
    p_embed, p_mask = premise.embeddings, premise.mask

    if self.map_embed is not None:
      with tf.variable_scope("map-embed"):
        q_embed = self.map_embed.apply(is_train, q_embed, q_mask)
      with tf.variable_scope("map-embed", reuse=True):
        p_embed = self.map_embed.apply(is_train, p_embed, p_mask)

    with tf.variable_scope("fuse"):
      fused = self.fuse_layer.apply(is_train, p_embed, q_embed, p_mask, q_mask)

    with tf.variable_scope("post-process-fused"):
      fused = self.post_process_layer.apply(is_train, fused, p_mask)

    logits = ops.affine(fused, 2, "predict-w")

    if labels is not None and "bias" in features:
      loss = self.debias_loss_fn.compute_qa_loss(
        q_embed, fused, logits, features["bias"], labels["answer_tokens"], p_mask)
      tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    return ops.mask_logits(logits, p_mask)
