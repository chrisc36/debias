import numpy as np
import tensorflow as tf

from debias.utils import ops

BOW_CHAR = 255
EOW_CHAR = 256
PAD_CHAR = 257
NUM_CHARS = PAD_CHAR+1


def word_to_char_ids(word, word_len):
  char_ids = tf.to_int32(tf.decode_raw(word, tf.uint8)[:word_len-2])
  padding = tf.fill([word_len - tf.shape(char_ids)[0] - 2], PAD_CHAR)
  char_ids = tf.concat([[BOW_CHAR], char_ids, [EOW_CHAR], padding], 0)
  char_ids.set_shape([word_len])
  return char_ids


def words_to_char_ids_py(words, word_length):
  out = np.full((len(words), word_length), PAD_CHAR, np.int32)
  out[:, 0] = BOW_CHAR
  for i, word in enumerate(words):
    word = list(word)[:word_length-2]
    out[i, 1:len(word)+1] = word
    out[i, len(word)+1] = EOW_CHAR
  return out


def words_to_char_ids(words, word_length, cpu=True):
  flat_words = tf.reshape(words, [-1])
  # Surprisingly, using a py_func here is much faster then the pure tensorflow option
  # Presumably because we have to .map in the tensorflow version which is very slow
  flat_char_ids = tf.py_func(
    lambda x: words_to_char_ids_py(x, word_length),
    [flat_words],
    [tf.int32],
    False
  )
  return tf.reshape(flat_char_ids, ops.get_shape_tuple(words) + [word_length])
  # if cpu:
  #   with tf.device("/cpu:0"):
  #     flat_char_ids = tf.map_fn(
  #       fn=lambda x: word_to_char_ids(x, word_length),
  #       elems=flat_words,
  #       dtype=tf.int32,
  #       back_prop=False
  #     )
  # else:
  #   flat_char_ids = tf.map_fn(
  #     fn=lambda x: word_to_char_ids(x, word_length),
  #     elems=flat_words,
  #     dtype=tf.int32,
  #     back_prop=False
  #   )
  #
  # return tf.reshape(flat_char_ids, ops.get_shape_tuple(words) + [word_length])


def embed_ids(cids, char_embed_dim):
  c_embed = tf.get_variable("char_embeddings", [NUM_CHARS, char_embed_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
  return tf.nn.embedding_lookup(c_embed, cids)
