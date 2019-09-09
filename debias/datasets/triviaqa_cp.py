import logging
from os.path import join, exists
from typing import List, Tuple, Set, Dict

import numpy as np
import tensorflow as tf

from debias import config
from debias.config import TRIVIAQA_CP_LOCATION_FILTERED_BIAS, TRIVIAQA_CP_PERSON_FILTERED_BIAS
from debias.datasets.dataset_utils import build_epoch_fn, build_stratified_epoch_fn
from debias.datasets.training_data_loader import HYPOTHESIS_KEY, PREMISE_KEY, TrainingDataLoader, \
  TrainingData
from debias.utils import py_utils
from triviaqa_cp import triviaqa_cp_loader

TRIVIAQA_CP_BIAS_FILE_IDS = {
  ("person", "train"): "1Oy-Lz8rdVY4mb80XzbrlZK3JHFu8TXt8",
  ("person", "dev"): "1gQILiHGaXM446hooFY84CxfkZsrtFlcb",
  ("location", "train"): "1Nu0s27RINvRVLo_jd4GBX-3cwcRcHnwV",
  ("location", "dev"): "1-NSQPSEcuUBri7P9PDVBiv8k7QnOSYc6",
}


TRIVIAQA_CP_CORENLP_FILE_IDS = {
  "train": "1AmnNqCY6a_agQvJlDjVZXMyjQEmrvtVB",
  "dev": "16VP_NjA_NcYwPbYgetutlxYX0Dv-qIFp"
}


TRIVIAQA_CP_FILE_IDS = {
  "train": "1Qjfpyb-Y2cvwmiT7tsBQF_LqC-rWVskn",
  "dev": "1mNt2GvXrra5EKmfHkQBMwpWuZAtiQ0am"
}


class AnnotatedTriviaQaExample:
  """TriviaQA Example that has been tokenized and POS/NER tagged"""

  def __init__(self, question_id: str, question_type: str,
               question_type_probs: np.ndarray,
               question: List[str], tokens: List[str],
               pos: List[str], ner: List[str],
               answers: List[str], answer_spans: np.ndarray):
    self.question_type = question_type
    self.question_type_probs = question_type_probs
    self.question_id = question_id
    self.question = question
    self.tokens = tokens
    self.ner = ner
    self.pos = pos
    self.answers = answers
    self.answer_spans = answer_spans


def load_annotated_triviaqa(is_train: bool) -> List[AnnotatedTriviaQaExample]:
  """Loads TriviaQA data that has been tokenized and tagged by CoreNLP"""
  dataset_name = "train" if is_train else "dev"
  src = join(config.TRIVIAQA_CP_CORENLP, "%s.pkl" % dataset_name)
  if not exists(src):
    logging.info("Download pre-processed TriviaQA %s to %s" % (dataset_name, src))
    py_utils.download_from_drive(TRIVIAQA_CP_CORENLP_FILE_IDS[dataset_name], src, progress_bar=True)

  logging.info("Loading CoreNLP TriviaQA %s..." % dataset_name)
  return py_utils.load_pickle(src)


def load_annotated_triviaqa_cp(dataset_name: str, part: str) -> List[AnnotatedTriviaQaExample]:
  """Loads TriviaQA-CP data that has been tokenized and tagged by CoreNLP"""
  triviaqa = load_annotated_triviaqa(part == "train")
  target_qytpes = triviaqa_cp_loader.get_qtypes(dataset_name, part)
  return [x for x in triviaqa if x.question_type in target_qytpes]


def load_bias(dataset_name: str, is_train=True) -> Dict[str, np.ndarray]:
  """Loads the output of our bias-only model

  Note that since this produces per-token output, it is only valid on data with the
  same tokenization as our annotated data.
  """
  if dataset_name == "location":
    cache_dir = TRIVIAQA_CP_LOCATION_FILTERED_BIAS
  elif dataset_name == "person":
    cache_dir = TRIVIAQA_CP_PERSON_FILTERED_BIAS
  else:
    raise ValueError(dataset_name)

  part_name = "train" if is_train else "dev"
  src = join(cache_dir, "%s.pkl" % part_name)

  if not exists(src):
    key = (dataset_name, part_name)
    if key not in TRIVIAQA_CP_BIAS_FILE_IDS:
      raise RuntimeError()
    logging.info("Downloading TriviaQA-CP bias for %s to %s" % (dataset_name, src))
    py_utils.download_from_drive(TRIVIAQA_CP_BIAS_FILE_IDS[key], src, progress_bar=False)

  return py_utils.load_pickle(src)


def load_triviaqa_cp(dataset_name: str, part: str) -> List[Dict]:
  """Load the official TriviaQA-CP dataset, needed for evaluation"""
  src_name = "train" if (part == "train") else "dev"
  src = join(config.TRIVIAQA_CP_SOURCE, src_name + ".json")
  if not exists(src):
    logging.info("Download TriviaQA-CP %s to %s" % (src_name, src))
    py_utils.download_from_drive(TRIVIAQA_CP_FILE_IDS[src_name], src, True)

  return triviaqa_cp_loader.load_triviaqa_cp(src, dataset_name, part)


def compute_voc(*datasets: List[AnnotatedTriviaQaExample]) -> Set[str]:
  voc = set()
  for data in datasets:
    for x in data:
      voc.update(x.tokens)
      voc.update(x.question)
  return voc


def convert_to_tuples(examples: List[AnnotatedTriviaQaExample]) -> List[Tuple]:
  out = []
  for ex in examples:
    dense_answers = np.zeros((len(ex.tokens), 2), np.bool)
    for s, e in ex.answer_spans:
      dense_answers[s, 0] = True
      dense_answers[e, 1] = True

    out.append((ex.question_id, ex.question, ex.tokens, dense_answers, ex.answers))
  return out


# To keep a consistent format with MNLI, use HYPOTHESIS/PREMISE to mean question/passage
base_features = [
  ("question_id", tf.string, ()),
  (HYPOTHESIS_KEY, tf.string, (None,)),  # The question
  (PREMISE_KEY, tf.string, (None,))  # The passage
]

label_structure = [
  ("answer_tokens", tf.bool, (None, 2)),
  ("answers", tf.string, (None, )),
]
n_label_elements = len(label_structure)


def make_dataset(lst: List[Tuple], bias=None, sample=None, shuffle=False) -> tf.data.Dataset:
  """Convert tuples from `convert_to_tuples` into a tf.data.Dataset"""
  dataset_structure = list(base_features)
  if bias:
    n = len(base_features)
    lst = [x[:n] + (bias[x[0]], ) + x[n:] for x in lst]
    dataset_structure.append(("bias", tf.float32, (None, 2)))

  dataset_structure += label_structure

  ds_names, ds_dtypes, ds_shapes = [tuple(x) for x in py_utils.transpose_lists(dataset_structure)]

  get = build_epoch_fn(lst, sample, shuffle)
  data = tf.data.Dataset.from_generator(get, ds_dtypes, ds_shapes)

  def to_dict(*args):
    labels = {k: v for k, v in zip(ds_names[-n_label_elements:], args[-n_label_elements:])}
    features = {k: v for k, v in zip(ds_names[:-n_label_elements], args[:-n_label_elements])}
    features["label"] = labels
    return features

  return data.map(to_dict)


def make_dataset_stratify(lst: List[Tuple], bias, n_groups) -> tf.data.Dataset:
  """Convert tuples from `convert_to_tuples` into a tf.data.Dataset,
  while stratifying on the bias accuracy"""
  dataset_structure = list(base_features)
  if bias:
    n = len(base_features)
    lst = [x[:n] + (bias[x[0]], ) + x[n:] for x in lst]
    dataset_structure.append(("bias", tf.float32, (None, 2)))

  dataset_structure += label_structure

  ds_names, ds_dtypes, ds_shapes = [tuple(x) for x in py_utils.transpose_lists(dataset_structure)]

  bias_ix = [i for i, name in enumerate(ds_names) if name == "bias"]
  if len(bias_ix) != 1:
    raise ValueError()
  bias_ix = bias_ix[0]

  bias_probs = []
  for example in lst:
    bias = example[bias_ix]
    spans = example[-2]
    if len(spans) == 0:
      bias_probs.append(0)
    else:
      valid = example[-2]
      bias_probs.append(bias[valid].sum())

  ix = np.argsort(bias_probs)
  lst = [lst[i] for i in ix]

  fn = build_stratified_epoch_fn(lst, n_groups)

  lst = tf.data.Dataset.from_generator(fn, ds_dtypes, ds_shapes)

  def to_dict(*args):
    labels = {k: v for k, v in zip(ds_names[-n_label_elements:], args[-n_label_elements:])}
    features = {k: v for k, v in zip(ds_names[:-n_label_elements], args[:-n_label_elements])}
    features["label"] = labels
    return features

  return lst.map(to_dict)


class AnnotatedTriviaQACPLoader(TrainingDataLoader):
  """TriviaQA-CP training data loader"""

  def __init__(self,
               dataset_name, load_bias=True,
               sample_train_eval=None, sample_train=None,
               sample_dev=None, stratify=False):
    if dataset_name not in ["person", "location"]:
      raise ValueError()
    self.dataset_name = dataset_name
    self.load_bias = load_bias
    self.sample_train = sample_train
    self.sample_train_eval = sample_train_eval
    self.sample_dev = sample_dev
    self.stratify = stratify

  def load(self, tokenizer, n_processes=None):
    train = load_annotated_triviaqa_cp(self.dataset_name, "train")
    dev = load_annotated_triviaqa_cp(self.dataset_name, "dev")

    if self.sample_train:
      train = np.random.choice(train, self.sample_train, False).tolist()
    if self.sample_dev:
      dev = np.random.choice(dev, self.sample_dev, False).tolist()

    voc = compute_voc(train, dev)

    train.sort(key=lambda x: len(x.tokens))
    train_tuples = convert_to_tuples(train)

    dev.sort(key=lambda x: len(x.tokens))
    dev_tuples = convert_to_tuples(dev)

    eval_sets = dict()

    if load_bias:
      bias = load_bias(self.dataset_name, True)
    else:
      bias = None

    if self.stratify:
      train = make_dataset_stratify(train_tuples, bias, self.stratify)
    else:
      train = make_dataset(train_tuples, bias, shuffle=True)

    if self.sample_train_eval:
      eval_sets["train"] = make_dataset(train_tuples, None, self.sample_train_eval, shuffle=False)
    eval_sets["dev"] = make_dataset(dev_tuples, None, None, shuffle=False)

    return TrainingData(train, eval_sets, voc)
