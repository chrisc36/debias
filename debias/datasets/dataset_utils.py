import logging

import numpy as np
import tensorflow as tf

from debias.datasets.training_data_loader import PREMISE_KEY
from debias.utils import py_utils, configured, ops


def build_epoch_fn(lst, sample=None, shuffle=False):
  """Build a function to return `lst` after sampling/shuffling"""
  if sample:
    def get():
      ix = np.random.choice(len(lst), sample, replace=False)
      if not shuffle:
        ix.sort()
      return [lst[i] for i in ix]

  elif shuffle:
      def get():
        cpy = list(lst)
        np.random.shuffle(cpy)
        return cpy
  else:
    get = lambda: lst

  return get


def build_stratified_epoch_fn(lst, n_groups):
  """Build a function to return `lst` after doing a stratified shuffle

  Assuming the data is sorted by a per-example score, the data will yield examples
  with scores that are deliberately spread out

  We used this for some of the QA dataset so its preserved here for exactness,
  although I *think* it doesn't really make a difference
  """

  # Split lst into group, assuming lst is sorted by the score we are stratifying on,
  # each group will contain examples with a similar score
  groups = py_utils.split(lst, n_groups)

  def build():
    local_groups = [list(x) for x in groups]
    for group in local_groups:
      # Shuffle the individual groups
      np.random.shuffle(group)

    # Merge the groups
    out = []
    while local_groups:
      for group in local_groups:
        out.append(group.pop())
      local_groups = [x for x in local_groups if len(x) > 0]

    return out

  return build


class QuantileBatcher(configured.Configured):
  """Batch a dataset by keeping a histogram of example lengths, and
  batching together examples whose lengths are in the same quantile"""

  def __init__(self, batch_size, hist_min, hist_max, hist_step, n_buckets):
    self.batch_size = batch_size
    self.hist_min = hist_min
    self.hist_max = hist_max
    self.hist_step = hist_step
    self.n_buckets = n_buckets

  def batch(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    bounds = list(range(self.hist_min, self.hist_max, self.hist_step))

    logging.info(
        "Quantile bucketing from %d-%d with %d buckets" %
        (bounds[0], bounds[-1], len(bounds)))

    return dataset.apply(ops.bucket_by_quantiles(
        len_fn=lambda x: tf.shape(x[PREMISE_KEY])[0],
        batch_size=self.batch_size,
        n_buckets=self.n_buckets,
        hist_bounds=bounds
    ))

