import logging
from os.path import join, exists
from typing import List, Tuple, Set, Dict

import numpy as np
import tensorflow as tf

from debias import config
from debias.config import SQUAD_TFIDF_FILTERED_BIAS, SQUAD_TFIDF_BIAS
from debias.datasets.dataset_utils import build_epoch_fn, build_stratified_epoch_fn
from debias.datasets.training_data_loader import TrainingData, TrainingDataLoader, HYPOTHESIS_KEY, \
  PREMISE_KEY
from debias.utils import py_utils

DATASETS = ["train", "dev", "add_sent", "add_one_sent"]

SQUAD_URLS = {
  "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
  "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
  "add_sent": "https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/",
  "add_one_sent": "https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/"
}


ANNOTATED_SQUAD_FILE_IDS = {
  "dev": "1j9lLz7jWg8F04iMM4C-iSFosq_33PWZe",
  "train": "129cF4sv8gws1hWBWAVAqlJ91_KBu4d54",
  "add_sent": "1OsSQTjo--SaI-lkiPVLp_x-1o7DCzFV3",
  "add_one_sent": "1BAjuKWrkx73y-wR-vfFnCj3tvAlhEWYK"
}


SQUAD_FILTERED_BIAS_FILE_IDS = {
  "train":  "1pi5-_OrGRD_HogYpNjoBoF2WauWVxofW",
  "dev": "1gZHw5rFfnJ4nih4E1qSIrOdJa_Lm7ZQj"
}

SQUAD_BIAS_FILE_IDS = {
  "train": "1Me6YyXuLQ8ZVhEK44Lv-4QxzD3dkGLc-",
  "dev": "14Cpulr3N2_C3gmIQxHk-CGxfmzB7_ZWZ"
}


class SquadQuestion:
  """Question and answers from the SQuAD dataset"""
  def __init__(self, question_id: str, words: List[str],
               answer_text: List[str],
               answer_spans: np.ndarray):
    self.question_id = question_id
    self.words = words
    self.answer_text = answer_text
    self.answer_spans = answer_spans

  def __repr__(self) -> str:
    return " ".join(self.words)


class AnnotatedSquadParagraph:
  """Tokenized paragraph from the SQuAD dataset, with POS/NER/Sentence annotations"""

  def __init__(self, passage_str, tokens: List[str], inv, pos_tags: List[str],
               ner_tags: List[str], sentence_lens, questions: List[SquadQuestion]):
    self.passage_str = passage_str
    self.tokens = tokens
    self.inv = inv
    self.pos_tags = pos_tags
    self.ner_tags = ner_tags
    self.sentence_lens = sentence_lens
    self.questions = questions

  def sentences(self):
    on = 0
    sentences = []
    for s in self.sentence_lens:
      sentences.append(self.tokens[on:on+s])
      on += s
    return sentences


def load_squad_documents(dataset_name) -> Dict:
  """Loads the original SQuAD data, needed to run the official evaluation scripts"""

  if dataset_name not in SQUAD_URLS:
    raise ValueError(dataset_name)
  src = join(config.SQUAD_SOURCE, dataset_name + ".json")
  if not exists(src):
    logging.info("Download SQuAD %s to %s" % (dataset_name, src))
    py_utils.download_to_file(SQUAD_URLS[dataset_name], src)

  return py_utils.load_json(src)["data"]


def load_annotated_squad(dataset_name) -> List[AnnotatedSquadParagraph]:
  """Loads SQuAD data that has been tokenized and tagged by CoreNLP"""

  if dataset_name not in DATASETS:
    raise ValueError("Invalid dataset %s" % dataset_name)
  src = join(config.SQUAD_CORENLP, "%s.pkl" % dataset_name)
  if not exists(src):
    logging.info("Download pre-processed SQuAD %s to %s" % (dataset_name, src))
    py_utils.download_from_drive(ANNOTATED_SQUAD_FILE_IDS[dataset_name], src)
  return py_utils.load_pickle(src)


def load_bias(dataset_name, filtered=False) -> Dict[str, np.ndarray]:
  """Loads the output of our bias-only model

  Note that since this produces per-token output, it is only valid on data with the
  same tokenization as our annotated data.
  """
  if filtered:
    bias_ids = SQUAD_FILTERED_BIAS_FILE_IDS
    output_dir = SQUAD_TFIDF_FILTERED_BIAS
  else:
    bias_ids = SQUAD_BIAS_FILE_IDS
    output_dir = SQUAD_TFIDF_BIAS

  if dataset_name not in bias_ids:
    raise ValueError("No bias for %s" % dataset_name)
  src = join(output_dir, "%s.pkl" % dataset_name)
  if not exists(src):
    logging.info("Downloading SQuAD bias for %s to %s" % (dataset_name, src))
    py_utils.download_from_drive(bias_ids[dataset_name], src)
  return py_utils.load_pickle(src)


def compute_voc(*datasets: List[AnnotatedSquadParagraph]) -> Set[str]:
  voc = set()
  for data in datasets:
    for x in data:
      voc.update(x.tokens)
      for q in x.questions:
        voc.update(q.words)
  return voc


def convert_to_tuples(examples: List[AnnotatedSquadParagraph]) -> List[Tuple]:
  """Convert SQuAD paragraphs to individual examples in tuple form"""
  tuples = []
  for ex in examples:
    for q in ex.questions:
      dense_answers = np.zeros((len(ex.tokens), 2), np.bool)
      for s, e in q.answer_spans:
        dense_answers[s, 0] = True
        dense_answers[e, 1] = True

      label = (ex.passage_str, ex.inv, dense_answers, q.answer_text)
      features = (q.question_id, q.words, ex.tokens)
      tuples.append(features + label)
  return tuples


# To keep a consistent format with MNLI, use HYPOTHESIS/PREMISE to mean question/passage
base_features = [
  ("question_id", tf.string, ()),
  (HYPOTHESIS_KEY, tf.string, (None,)),  # The question
  (PREMISE_KEY, tf.string, (None,))  # The passage
]

# For SQuAD we keep track of extra information so we can 'untokenize' the passage, which can
# be important to get a good score
label_structure = [
  ("passage_str", tf.string, ()),  # Original passage string
  ("token_offsets", tf.int32, (None, 2)),  # Character spans for each token
  ("answer_tokens", tf.bool, (None, 2)),  # Dense start/end answer tokens
  ("answers", tf.string, (None, )),  # List of string answers
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
  while stratifying the bias accuracy"""
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


class AnnotatedSquadLoader(TrainingDataLoader):
  """SQuAD training data loader"""

  def __init__(self,
               load_bias=True,
               filtered_bias=True,
               sample_train_eval=None, sample_dev_eval=None,
               sample_train=None, sample_dev=None,
               max_train_len=None,
               stratify=False):
    self.filtered_bias = filtered_bias
    self.load_bias = load_bias
    self.sample_train = sample_train
    self.max_train_len = max_train_len
    self.sample_dev = sample_dev
    self.sample_train_eval = sample_train_eval
    self.sample_dev_eval = sample_dev_eval
    self.stratify = stratify

  def load(self, tokenizer, n_processes=None):
    targets = ["train", "dev"]

    datasets = []
    for eval_set in targets:
      tf.logging.info("Loading %s..." % eval_set)
      data = load_annotated_squad(eval_set)

      if eval_set == "train" and self.sample_train:
        data = np.random.choice(data, self.sample_train, False).tolist()
      elif eval_set == "dev" and self.sample_dev:
        data = np.random.choice(data, self.sample_dev, False).tolist()

      datasets.append(data)

    voc = compute_voc(*datasets)

    tuple_datasets = []
    for name, examples in zip(targets, datasets):
      examples.sort(key=lambda x: len(x.tokens))
      tuple_datasets.append(convert_to_tuples(examples))

    eval_sets = dict()
    train_tuples, dev_tuples = tuple_datasets

    if self.load_bias:
      bias = load_bias("train", True)
    else:
      bias = None

    if self.stratify:
      train = make_dataset_stratify(train_tuples, bias, self.stratify)
    else:
      train = make_dataset(train_tuples, bias, shuffle=True)

    if self.sample_train_eval:
      eval_sets["train"] = make_dataset(train_tuples, None, self.sample_train_eval, shuffle=False)
    eval_sets["dev"] = make_dataset(dev_tuples, None, self.sample_dev, shuffle=False)
    return TrainingData(train, eval_sets, voc)

