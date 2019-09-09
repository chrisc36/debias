from typing import List
from zlib import crc32

import numpy as np

from debias.datasets.mnli import TextPairExample, tokenize_examples, load_mnli, make_dataset
from debias.datasets.training_data_loader import TrainingDataLoader, TrainingData


def add_noise(examples: List[TextPairExample], prob_correct, indicator_prob, n_classes, build_bias_model=False):
  """
  :param examples: Tokenized TextPair examples to modify
  :param prob_correct: Chance the bias token agrees with the label
  :param indicator_prob: Optionally, chance the indicator token is one, in which case the bias will be random
  :param n_classes:
  :param build_bias_model: Build and return a function that maps  TextPairExample -> bias predictions
  :return: List of modified `TextPairExample`, and optionally the bias-only model
  """
  example_noise = []
  indicators = []
  for x in examples:
    if isinstance(x.hypothesis, str):
      raise ValueError("Example not tokenized")
    l = x[3]
    # To ensure the noise is consistent between examples, use a hash as the seed
    h = crc32((str(x.label) + "\n" + " ".join(x.hypothesis) + "\n" + " ".join(x.premise)).encode("utf-8"))
    rng = np.random.RandomState(h)

    if indicator_prob is not None:
      if rng.uniform(0, 1) < indicator_prob:
        # Indicator is off, the bias matches with `prob_correct`
        group = 0
        p = prob_correct
      else:  # Indicator is set, the bias is random
        group = 1
        p = 1/n_classes
      indicators.append(group)

      match_label = rng.uniform(0, 1) < p
    else:
      match_label = rng.uniform(0, 1) < prob_correct

    if match_label:
      noise = l
    else:
      noise = (l + rng.randint(1, n_classes)) % n_classes  # Select a different class
    example_noise.append(noise)

  out = []
  for i, x in enumerate(examples):
    n = example_noise[i]
    if indicator_prob is not None:
      new_hypothesis = [str(n)] + x.hypothesis + [str(indicators[i])]
    else:
      new_hypothesis = [str(n)] + x.hypothesis
    out.append(TextPairExample(x.id, x.premise, new_hypothesis, x.label))

  if build_bias_model:
    # 'Trains' the bias-only model, to simulate the fact the bias-only model will not be perfect
    # it uses the empirical bias/label correlation, not the true correlation
    bias_model = np.zeros((3, 3))
    counts = np.zeros(3)
    for x, n in zip(examples, example_noise):
      bias_model[n, x.label] += 1
      counts[n] += 1

    bias_model /= np.expand_dims(counts, 1)
    bias_model = np.log(bias_model).astype(np.float32)

    def compute_bias(ex: TextPairExample):
      return bias_model[int(ex.hypothesis[0])]

    return out, compute_bias
  else:
    return out


class MnliWithSyntheticBiasLoading(TrainingDataLoader):
  """Loader for our synthetic experiments"""

  def __init__(self,
               train_noise, n_train_eval, indicator_noise=None, n_train_sample=None,
               n_dev_sample=None):
    self.train_noise = train_noise
    self.n_train_eval = n_train_eval
    self.n_train_sample = n_train_sample
    self.n_dev_sample = n_dev_sample
    self.indicator_noise = indicator_noise

  def load(self, tokenizer, n_processes=None):
    train = tokenize_examples(load_mnli(True, self.n_train_sample), tokenizer, n_processes)
    dev = tokenize_examples(load_mnli(False, self.n_dev_sample), tokenizer, n_processes)
    dev.sort(key=lambda x: len(x.premise))
    train.sort(key=lambda x: len(x.premise))

    voc = set()
    for _, p, h, _ in train:
      voc.update(p)
      voc.update(h)
    for _, p, h, _ in dev:
      voc.update(p)
      voc.update(h)

    train, bias_fn = add_noise(train, self.train_noise, self.indicator_noise, 3, True)
    train_bias = {x.id: bias_fn(x) for x in train}

    train_ds = make_dataset(train, train_bias)

    if self.n_train_eval > 0:
      train_eval = make_dataset(train, None, sample=self.n_train_eval, shuffle=False)

    dev = add_noise(dev, self.train_noise, self.indicator_noise, 3, False)
    dev.sort(key=lambda x: len(x.premise))
    dev_eval = make_dataset(dev, None, sample=self.n_dev_sample, shuffle=False)

    eval_sets = dict(biased_dev=dev_eval, train=train_eval)
    return TrainingData(train_ds, eval_sets, voc)
