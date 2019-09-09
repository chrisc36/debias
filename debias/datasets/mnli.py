"""Methods to load HANS and MNLI examples, tokenize them, and use them to build a tf.data.Dataset"""
import logging
from collections import namedtuple
from os import listdir
from os.path import join, exists
from typing import List, Dict, Optional, Iterable

import numpy as np
import tensorflow as tf

from debias import config
from debias.datasets.dataset_utils import build_epoch_fn
from debias.datasets.training_data_loader import TrainingDataLoader, TrainingData, HYPOTHESIS_KEY, \
  PREMISE_KEY
from debias.utils import py_utils, process_par

TextPairExample = namedtuple("TextPairExample", ["id", "premise", "hypothesis", "label"])

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
# Taken from the GLUE script
MNLI_URL = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce"

MNLI_BIAS_DRIVE_IDS = {
  "train": "1ctP0T93F02IjWh4m1CMUdrSRQnoxEg6f",
  "dev": "10PcUMd6xqXArwRPCkIYjIwdL59f2p5TO",
  "hans": "1DEQaUmwfTgK6kSHqXj5phySPYwUq-52S",
}

NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}


def load_hans(n_samples=None) -> List[TextPairExample]:
  out = []
  logging.info("Loading hans...")
  src = join(config.HANS_SOURCE, "heuristics_evaluation_set.txt")
  if not exists(src):
    logging.info("Downloading source to %s..." % config.HANS_SOURCE)
    py_utils.download_to_file(HANS_URL, src)

  with open(src, "r") as f:
    f.readline()
    lines = f.readlines()

  if n_samples is not None:
    lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples, replace=False)

  for line in lines:
    parts = line.split("\t")
    label = parts[0]
    if label == "non-entailment":
      label = 0
    elif label == "entailment":
      label = 1
    else:
      raise RuntimeError()
    s1, s2, pair_id = parts[5:8]
    out.append(TextPairExample(pair_id, s1, s2, label))
  return out


def ensure_mnli_is_downloaded():
  mnli_source = join(config.GLUE_SOURCE, "MNLI")
  if exists(mnli_source) and len(listdir(mnli_source)) > 0:
    return
  py_utils.download_zip("MNLI", MNLI_URL, config.GLUE_SOURCE)


def load_mnli(is_train, sample=None) -> List[TextPairExample]:
  ensure_mnli_is_downloaded()
  if is_train:
    filename = join(config.GLUE_SOURCE, "MNLI", "train.tsv")
  else:
    filename = join(config.GLUE_SOURCE, "MNLI", "dev_matched.tsv")

  logging.info("Loading mnli " + ("train" if is_train else "dev"))
  with open(filename) as f:
    f.readline()
    lines = f.readlines()

  if sample:
    lines = np.random.RandomState(26096781 + sample).choice(lines, sample, replace=False)

  out = []
  for line in lines:
    line = line.split("\t")
    out.append(TextPairExample(line[0], line[8], line[9], NLI_LABEL_MAP[line[-1].rstrip()]))
  return out


def load_bias(dataset_name) -> Dict[str, np.ndarray]:
  """Load dictionary of example_id->bias where bias is a length 3 array
  of log-probabilities"""

  if dataset_name not in MNLI_BIAS_DRIVE_IDS:
    raise ValueError(dataset_name)
  bias_src = join(config.MNLI_WORD_OVERLAP_BIAS, dataset_name + ".pkl")
  if not exists(bias_src):
    logging.info("Downloading MNLI bias to %s..." % bias_src)
    py_utils.download_from_drive(MNLI_BIAS_DRIVE_IDS[dataset_name], bias_src)

  bias = py_utils.load_pickle(bias_src)
  for k, v in bias.items():
    # Convert from entail vs non-entail to 3-way classes by splitting non-entail
    # to neutral and contradict
    bias[k] = np.array([
      v[0] - np.log(2.),
      v[1],
      v[0] - np.log(2.),
    ])
  return bias


class TokenizeProcessor(process_par.Processor):
  """For tokenizing examples in parallel"""

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

  def process(self, data: Iterable[TextPairExample]) -> List[TextPairExample]:
    tokenizer = self.tokenizer
    out = []
    for example in data:
      out.append(TextPairExample(
        example.id,
        tokenizer.tokenize(example.premise),
        tokenizer.tokenize(example.hypothesis),
        example.label
      ))
    return out


def tokenize_examples(examples: List[TextPairExample], tokenizer, n_processes) -> List[TextPairExample]:
  return process_par.process_par(
    examples, TokenizeProcessor(tokenizer), n_processes, desc="tokenizing")


def make_dataset(data: List[TextPairExample], bias: Optional[Dict]=None, sample=None, shuffle=True) -> tf.data.Dataset:
  if bias:
    data = [tuple(x) + (bias[x.id], ) for x in data]

  fn = build_epoch_fn(data, sample, shuffle=shuffle)
  structure = [
    ("id", tf.string, ()),
    (PREMISE_KEY, tf.string, (None, )),
    (HYPOTHESIS_KEY, tf.string, (None,)),
    ("label", tf.int32, ())
  ]
  if bias:
    structure.append(("bias", tf.float32, (3, )))
  names, dtypes, shapes = py_utils.transpose_lists(structure)
  ds = tf.data.Dataset.from_generator(fn, tuple(dtypes), tuple(shapes))

  def to_map(*args):
    return {k: v for k, v in zip(names, args)}

  return ds.map(to_map)


class MnliTrainingDataLoader(TrainingDataLoader):

  def __init__(self, n_train_eval_sample, n_train_sample=None, n_dev_sample=None, use_bias=True):
    self.n_train_eval_sample = n_train_eval_sample
    self.n_train_sample = n_train_sample
    self.use_bias = use_bias
    self.n_dev_sample = n_dev_sample

  def load(self, tokenizer, n_processes=None):
    train = tokenize_examples(load_mnli(True, self.n_train_sample), tokenizer, n_processes)
    dev = tokenize_examples(load_mnli(False, self.n_dev_sample), tokenizer, n_processes)
    dev.sort(key=lambda x: len(x.premise))
    train.sort(key=lambda x: len(x.premise))

    voc = set()
    for ds in [train, dev]:
      for ex in ds:
        voc.update(ex.premise)
        voc.update(ex.hypothesis)

    if self.use_bias:
      bias = load_bias("train")
    else:
      bias = None

    train_ds = make_dataset(train, bias, shuffle=True)
    eval_sets = dict()
    if self.n_train_eval_sample:
      eval_sets["train"] = make_dataset(train, None, shuffle=False, sample=self.n_train_eval_sample)

    eval_sets["dev"] = make_dataset(dev, None, shuffle=False)

    return TrainingData(train_ds, eval_sets, voc)
