from typing import Optional

import tensorflow as tf

from debias.models.text_model import TextModel
from debias.modules.attention_layers import AttentionBiFuse
from debias.modules.clf_debias_loss_functions import ClfDebiasLossFunction
from debias.modules.layers import SequenceMapper, PoolingLayer, Mapper
from debias.modules.word_and_char_encoder import WordAndCharEncoder
from debias.utils import ops
from debias.utils.tokenizer import Tokenizer


class TextPairClfDebiasingModel(TextModel):
  def __init__(self,
               tokenizer: Tokenizer,
               text_encoder: WordAndCharEncoder,
               map_embed: Optional[SequenceMapper],
               bifuse_layer: AttentionBiFuse,
               post_process_layer: SequenceMapper,
               pool_layer: PoolingLayer,
               processs_joint: Mapper,
               n_classes,
               debias_loss_fn: ClfDebiasLossFunction
               ):
    super().__init__(tokenizer, text_encoder)
    self.map_embed = map_embed
    self.bifuse_layer = bifuse_layer
    self.pool_layer = pool_layer
    self.post_process_layer = post_process_layer
    self.processs_joint = processs_joint
    self.n_classes = n_classes
    self.debias_loss_fn = debias_loss_fn

  def apply(self, is_train, features, labels):
    hypoth, premise = self.get_text_embeddings(is_train, features)
    h_embed, h_mask = hypoth.embeddings, hypoth.mask
    p_embed, p_mask = premise.embeddings, premise.mask

    if self.map_embed is not None:
      with tf.variable_scope("map-embed"):
        h_embed = self.map_embed.apply(is_train, h_embed, h_mask)
      with tf.variable_scope("map-embed", reuse=True):
        p_embed = self.map_embed.apply(is_train, p_embed, p_mask)

    with tf.variable_scope("fuse"):
      p_fused, h_fused = self.bifuse_layer.apply(is_train, p_embed, h_embed, p_mask, h_mask)

    with tf.variable_scope("post-process-fused"):
      p_fused = self.post_process_layer.apply(is_train, p_fused, p_mask)

    with tf.variable_scope("post-process-fused", reuse=True):
      h_fused = self.post_process_layer.apply(is_train, h_fused, h_mask)

    with tf.variable_scope("pool"):
      p_pooled = self.pool_layer.apply(is_train, p_fused, p_mask)

    with tf.variable_scope("pool", reuse=True):
      h_pooled = self.pool_layer.apply(is_train, h_fused, h_mask)

    joint = tf.concat([p_pooled, h_pooled], 1)
    with tf.variable_scope("post-process-pooled"):
      joint = self.processs_joint.apply(is_train, joint)

    logits = ops.affine(joint, self.n_classes, "w", "b")
    if labels is not None and "bias" in features:
      loss = self.debias_loss_fn.compute_clf_loss(joint, logits, features["bias"], labels)
      tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    return logits
