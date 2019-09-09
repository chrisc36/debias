import logging

import numpy as np
import tensorflow as tf

from debias.modules import char_encoder
from debias.utils import load_word_vectors, ops
from debias.utils.configured import Configured


class WordAndCharEncoder(Configured):
  """Embed text using word vectors and character embeddings."""

  def __init__(self, word_vectors, first_n=None, char_embed_dim=None,
               character_mapper=None, character_pooler=None,
               lower_fallback=False,
               embed_cpu=True, word_pooling=True,
               word_mapper=None, lowercase=False, word_length=30,
               include_bounds_embeddings: bool=False):
    if character_mapper is not None and character_pooler is None:
      raise ValueError()
    if character_pooler is not None and character_mapper is None:
      raise ValueError()

    self.include_bounds_embeddings = include_bounds_embeddings
    self.lower_fallback = lower_fallback
    self.word_pooling = word_pooling
    self.character_mapper = character_mapper
    self.embed_cpu = embed_cpu
    self.char_embed_dim = char_embed_dim
    self.lowercase = lowercase
    self.character_pooler = character_pooler
    self.word_mapper = word_mapper
    self.word_vectors = word_vectors
    self.first_n = first_n
    self.word_length = word_length

    self._input_vocab = None
    self._cached_char_ids = None

    self._word_vec_vocab = None
    self._embeddings = None

  def set_vocab(self, vocab):
    self._cached_char_ids = None
    self._embeddings = None
    self._word_vec_vocab = None
    if vocab is None or not self.word_pooling:
      self._input_vocab = None
    elif self.lowercase:
      self._input_vocab = list(set(x.lower() for x in vocab))
    else:
      self._input_vocab = list(vocab)

  def get_vocab(self):
    return self._input_vocab

  @property
  def use_word_vecs(self):
    return self.word_vectors is not None

  @property
  def use_chars(self):
    return (self.char_embed_dim is not None and self.character_pooler is not None)

  def tensorize_fn(self):
    if (self._input_vocab is not None and
        self._cached_char_ids is None and self.use_chars):
      # Pre-compute the charids for all words in our vocab
      self._cached_char_ids = char_encoder.words_to_char_ids_py(
        [x.encode("utf-8") for x in self._input_vocab] + [""],
        self.word_length
      )

    if self._embeddings is None and self.use_word_vecs:
      # Load and cache the words vectors
      if self.lower_fallback:
        v = set(self._input_vocab)
        v.update(x.lower() for x in self._input_vocab)
        v.update(x[0].lower() + x[1:] for x in self._input_vocab)
        words_l, vecs_l = load_word_vectors.load_word_vectors(self.word_vectors, v, self.first_n)
        w_to_ix = {w: i for i, w in enumerate(words_l)}

        words = []
        vecs = []
        for w in self._input_vocab:
          ix = w_to_ix.get(w)
          if ix is None and w[0].isupper():
            ix = w_to_ix.get(w[0].lower() + w[1:])
          if ix is None:
            ix = w_to_ix.get(w.lower())
          if ix is not None:
            words.append(w)
            vecs.append(vecs_l[ix])

      else:
        words, vecs = load_word_vectors.load_word_vectors(self.word_vectors, self._input_vocab, self.first_n)

      if self._input_vocab is not None:
        logging.info("Have vectors for %d/%d (%.4f) words" % (
          len(words), len(self._input_vocab), len(words)/len(self._input_vocab)))

      dim = len(vecs[0])

      if self._input_vocab is None:
        self._embeddings = np.stack(vecs + [np.zeros(dim, dtype=np.float32)])
        self._word_vec_vocab = words
      else:
        w_to_ix = {w: i for i, w in enumerate(words)}
        self._embeddings = np.zeros((len(self._input_vocab)+1, dim), np.float32)
        for i, word in enumerate(self._input_vocab):
          ix = w_to_ix.get(word)
          if ix is not None:
            self._embeddings[i] = vecs[ix]
        self._word_vec_vocab = self._input_vocab + [""]

    elif self._input_vocab is not None:
      self._word_vec_vocab = self._input_vocab

    if self._word_vec_vocab is not None:
      tbl = tf.contrib.lookup.index_table_from_tensor(
        mapping=self._word_vec_vocab,
        num_oov_buckets=1 if self._input_vocab is None else 0
      )
    else:
      tbl = None

    def fn(string_tensor):
      """Builds the output tensor dictionary."""
      if self.lowercase:
        string_tensor = ops.lowercase_op(string_tensor)

      out = []
      if tbl is not None:
        wids = tf.to_int32(tbl.lookup(string_tensor))

        if self._input_vocab is not None:
          errors = tf.less(wids, 0)
          with tf.control_dependencies([tf.assert_greater_equal(
            tf.reduce_min(wids), 0,
            data=[tf.boolean_mask(string_tensor, errors)],
            summarize=100, message="Words missing from vocab"
          )]):
            wids = tf.identity(wids)
        out.append(wids)

      if self.use_chars and self._input_vocab is None:
        cids = char_encoder.words_to_char_ids(string_tensor, self.word_length)
        out.append(cids)

      return tuple(out)

    return fn

  def embed_words(self, is_train, tensors):
    if self._input_vocab is None:
      w_embeds = []
      for features in tensors:
        w_embeds.append(self._embed_words_from_features(is_train, features))
        tf.get_variable_scope().reuse_variables()
      return w_embeds
    else:
      return self._embed_words_from_ids(is_train, tensors)[0]

  def get_word_ids(self, tensors):
    if self._input_vocab is None:
      return None
    else:
      return tensors[0]

  def embed_words_with_ids(self, is_train, tensors, embed_words_with_ids=None):
    return self._embed_words_from_ids(is_train, tensors, embed_words_with_ids)

  def _embed_words_from_features(self, is_train, tensors):
    out = []
    if self.use_word_vecs:
      if self.embed_cpu:
        with tf.device("/cpu:0"):
          embeding_var = ops.as_initialized_variable(self._embeddings, "embeddings")
          w_embed = tf.nn.embedding_lookup(embeding_var, tensors[0])
      else:
        embeding_var = ops.as_initialized_variable(self._embeddings, "embeddings")
        w_embed = tf.nn.embedding_lookup(embeding_var, tensors[0])
      out = [w_embed]
      tensors = tensors[1:]

    if self.use_chars:
      cids = tensors[0]
      char_emb = char_encoder.embed_ids(cids, self.char_embed_dim)

      with tf.variable_scope("char-map"):
        char_emb = self.character_mapper.apply(is_train, char_emb)

      with tf.variable_scope("char-pool"):
        char_emb = self.character_pooler.apply(is_train, char_emb)
      out.append(char_emb)

    return tf.concat(out, -1)

  def _embed_words_from_ids(self, is_train, tensors, additional_wids=None):
    if self._input_vocab is None:
      raise NotImplementedError()
    wids = [x[0] for x in tensors]

    shapes = [ops.get_shape_tuple(x) for x in wids]
    unique_wids = [ops.flatten(x) for x in wids]
    sizes = [ops.get_shape_tuple(x, 0) for x in unique_wids]

    if additional_wids is not None:
      unique_wids.append(additional_wids)

    unique_wids = tf.concat(unique_wids, 0)
    wixs, w_mapping = tf.unique(ops.flatten(unique_wids), tf.int32)

    if additional_wids is not None:
      w_mapping = w_mapping[:-ops.get_shape_tuple(additional_wids)[0]]

    if self.use_word_vecs:
      if self.embed_cpu:
        with tf.device("/cpu:0"):
          embeding_var = ops.as_initialized_variable(self._embeddings, "embeddings")
          w_embed = [tf.nn.embedding_lookup(embeding_var, wixs)]
      else:
        embeding_var = ops.as_initialized_variable(self._embeddings, "embeddings")
        w_embed = [tf.nn.embedding_lookup(embeding_var, wixs)]
    else:
      w_embed = []

    if self.use_chars:
      with tf.device("/cpu:0"):
        cids = tf.gather(self._cached_char_ids, wixs)
        # dim = self._cached_char_ids.shape[1]
        # cids = tf.matmul(self._cached_char_ids,
        #                  tf.one_hot(wixs, dim, dtype=tf.int32))

      char_emb = char_encoder.embed_ids(cids, self.char_embed_dim)

      with tf.variable_scope("char-map"):
        char_emb = self.character_mapper.apply(is_train, char_emb)

      with tf.variable_scope("char-pool"):
        char_emb = self.character_pooler.apply(is_train, char_emb)

      w_embed.append(char_emb)

    w_embed = tf.concat(w_embed, 1)
    unique_word_embeddings = w_embed

    if self.word_mapper is not None:
      with tf.variable_scope("word-map"):
        w_embed = self.word_mapper.apply(is_train, w_embed)

    dim = w_embed.shape.as_list()[-1]

    # Undo tf.unique
    w_embeds = tf.gather(w_embed, w_mapping)

    # Undo tf.concat
    w_embeds = tf.split(w_embeds, sizes, 0)
    w_mapping = tf.split(w_mapping, sizes, 0)

    # Undo ops.flatten
    w_embeds = [tf.reshape(t, s + [dim]) for t, s in zip(w_embeds, shapes)]
    w_mapping = [tf.reshape(t, s) for t, s in zip(w_mapping, shapes)]
    return w_embeds, w_mapping, unique_word_embeddings
