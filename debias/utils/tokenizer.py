import logging
from typing import List, Tuple

import nltk
import numpy as np
import regex

from debias.utils.configured import Configured
from debias.utils.py_utils import flatten_list


class Tokenizer(Configured):
  def tokenize(self, text: str) -> List[str]:
    raise NotImplementedError()

  def tokenize_with_inverse(self, text: str) -> Tuple[List[str], np.ndarray]:
    """Tokenize the text, and return start/end character mapping of each token within `text`"""
    raise NotImplementedError()


_double_quote_re = regex.compile(u"\"|``|''")


def convert_to_spans(raw_text: str, text: List[str]) -> np.ndarray:
  """ Convert a tokenized version of `raw_text` into a series character
  spans referencing the `raw_text` """
  cur_idx = 0
  all_spans = np.zeros((len(text), 2), dtype=np.int32)
  for i, token in enumerate(text):
    if _double_quote_re.match(token):
      span = _double_quote_re.search(raw_text[cur_idx:])
      tmp = cur_idx + span.start()
      l = span.end() - span.start()
    else:
      tmp = raw_text.find(token, cur_idx)
      l = len(token)

    if tmp < cur_idx:
      raise ValueError(token)
    cur_idx = tmp
    all_spans[i] = (cur_idx, cur_idx + l)
    cur_idx += l
  return all_spans


class NltkAndPunctTokenizer(Tokenizer):
  """Tokenize ntlk, but additionally split on most punctuations symbols"""

  def __init__(self, split_dash=True, split_single_quote=False, split_period=False, split_comma=False):
    self.split_dash = split_dash
    self.split_single_quote = split_single_quote
    self.split_period = split_period
    self.split_comma = split_comma

    # Unix character classes to split on
    resplit = r"\p{Pd}\p{Po}\p{Pe}\p{S}\p{Pc}"

    # A list of optional exceptions, will we trust nltk to split them correctly
    # unless otherwise specified by the ini arguments
    dont_split = ""
    if not split_dash:
      dont_split += "\-"
    if not split_single_quote:
      dont_split += "'"
    if not split_period:
      dont_split += "\."
    if not split_comma:
      dont_split += ","

    resplit = "([" + resplit + "]|'')"
    if len(dont_split) > 0:
      split_regex = r"(?![" + dont_split + "])" + resplit
    else:
      split_regex = resplit

    self.split_regex = regex.compile(split_regex)
    try:
      self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')
    except LookupError:
      logging.info("Downloading NLTK punkt tokenizer")
      nltk.download('punkt')
      self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')

    self.word_tokenizer = nltk.TreebankWordTokenizer()

  def retokenize(self, x):
    if _double_quote_re.match(x):
      # Never split isolated double quotes(TODO Just integrate this into the regex?)
      return (x, )
    return (x.strip() for x in self.split_regex.split(x) if len(x) > 0)

  def tokenize(self, text: str) -> List[str]:
    out = []
    for s in self.sent_tokenzier.tokenize(text):
      out += flatten_list(self.retokenize(w) for w in self.word_tokenizer.tokenize(s))
    return out

  def tokenize_with_inverse(self, paragraph: str):
    text = self.tokenize(paragraph)
    inv = convert_to_spans(paragraph, text)
    return text, inv
