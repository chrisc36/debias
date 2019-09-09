from typing import Set, Dict

import tensorflow as tf

from debias.utils.configured import Configured

# Our tf.data.Dataset objects stores tokens with these keys
HYPOTHESIS_KEY = "hypothesis_tok"
PREMISE_KEY = "premise_tok"

# These should be added before batching, to preserve the length information
HYPOTHESIS_LEN_KEY = "hypothesis_tok_len"
PREMISE_LEN_KEY = "premise_tok_len"


class TrainingData:
  """Datasets we can train models with in `debias.training.Trainer`"""

  def __init__(self, train: tf.data.Dataset, eval_sets: Dict[str, tf.data.Dataset], voc: Set[str]):
    """
    All datasets yield individual examples, and should have (at least) the following structure:
    {
      HYPOTHESIS_KEY: <tf.string>[hypothesis_len] hypothesis or question tokens
      PREMISE_KEY: <tf.string>[premise_len] premise or passage tokens
      "label": label information, can be a tensor or dictionary
    }

    :param train: Training data, should yield shuffled examples
    :param eval_sets: Dictionary of eval sets, should yield sorted example
    :param voc: Set of words in all datasets
    """
    self.train = train
    self.eval_sets = eval_sets
    self.voc = voc


class TrainingDataLoader(Configured):
  def load(self, tokenizer, n_processes=None) -> TrainingData:
    raise NotImplementedError()
