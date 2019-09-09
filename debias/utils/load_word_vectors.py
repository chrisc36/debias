"""Loading words vectors."""
import gzip
import logging
import pickle
from os.path import join, exists
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from debias import config
from debias.utils import py_utils

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
GLOVE_6B_VECS = ["glove.6B.100d", "glove.6B.200d",
                 "glove.6B.300d",  "glove.6B.50d"]
GLOVE_6B_URL = "http://nlp.stanford.edu/data/glove.6B.zip"


def download_word_vectors(vec_name):
  if vec_name == "crawl-300d-2M":
    download_fasttext()
  elif vec_name in GLOVE_6B_VECS:
    download_glove_6b()
  else:
    raise NotImplementedError(
      vec_name + " does not exist, and cannot be automatically downloaded, please download manually")


def download_fasttext():
  if exists(join(config.WORD_VEC_SOURCE, "crawl-300d-2M.vec")):
    return
  py_utils.download_zip("crawl-300d-2M.vec", FASTTEXT_URL, config.WORD_VEC_SOURCE)


def download_glove_6b():
  if all(exists(join(config.WORD_VEC_SOURCE, x + ".txt")) for x in GLOVE_6B_VECS):
    return
  py_utils.download_zip("Glove 6B", GLOVE_6B_URL, config.WORD_VEC_SOURCE)


def _find_vec_path(vec_name):
  vec_path = join(config.WORD_VEC_SOURCE, vec_name)
  if exists(vec_path + ".txt"):
    return vec_path + ".txt"
  elif exists(vec_path + ".txt.gz"):
    return vec_path + ".txt.gz"
  elif exists(vec_path + ".pkl"):
    return vec_path + ".pkl"
  elif exists(vec_path + ".vec"):
    return vec_path + ".vec"
  else:
    return None


def load_word_vectors(vec_name: str, vocab: Optional[Iterable[str]]=None, n_words_to_scan=None):
  vec_path = _find_vec_path(vec_name)
  if vec_path is None:
    download_word_vectors(vec_name)
  vec_path = _find_vec_path(vec_name)
  if vec_path is None:
    raise RuntimeError("Download bug?")

  return load_word_vector_file(vec_path, vocab, n_words_to_scan)


def load_word_vector_file(vec_path: str, vocab: Optional[Iterable[str]]=None,
                          n_words_to_scan=None):
  if vocab is not None:
    vocab = set(vocab)

  if vec_path.endswith(".pkl"):
    with open(vec_path, "rb") as f:
      return pickle.load(f)

  # some of the large vec files produce utf-8 errors for some words, just skip them
  elif vec_path.endswith(".txt.gz"):
    handle = lambda x: gzip.open(x, 'r', encoding='utf-8', errors='ignore')
  else:
    handle = lambda x: open(x, 'r', encoding='utf-8', errors='ignore')

  if n_words_to_scan is None:
    if vocab is None:
      logging.info("Loading word vectors from %s..." % vec_path)
    else:
      logging.info("Loading word vectors from %s for voc size %d..." % (vec_path, len(vocab)))
  else:
    if vocab is None:
      logging.info("Loading up to %d word vectors from %s..." % (n_words_to_scan, vec_path))
    else:
      logging.info("Loading up to %d word vectors from %s for voc size %d..." % (n_words_to_scan, vec_path, len(vocab)))

  words = []
  vecs = []
  pbar = tqdm(desc="word-vec")
  with handle(vec_path) as fh:
    for i, line in enumerate(fh):
      pbar.update(1)
      if n_words_to_scan is not None and i >= n_words_to_scan:
        break
      word_ix = line.find(" ")
      if i == 0 and " " not in line[word_ix+1:]:
        # assume a header row, such as found in the fasttext word vectors
        continue
      word = line[:word_ix]
      if (vocab is None) or (word in vocab):
        words.append(word)
        vecs.append(np.fromstring(line[word_ix+1:], sep=" ", dtype=np.float32))

  pbar.close()
  return words, vecs
