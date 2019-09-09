import argparse
import json
import logging
import pickle
from os.path import join, exists
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from debias.datasets import mnli, synthetic
from debias.models.model_dir import ModelDir
from debias.utils import py_utils


def add_bias(all_examples, bias_name, unbaised):
  if bias_name == "indicator":
    bias_prob, i_prob = 0.8, None
  elif bias_name == "excluder":
    bias_prob, i_prob = 0.03, None
  elif bias_name == "dependent":
    bias_prob, i_prob = 0.9, 0.8
  else:
    raise RuntimeError()
  if unbaised:
    bias_prob = 1/3.0
  return synthetic.add_noise(all_examples, bias_prob, i_prob, 3)


def get_dataset_name(bias_name, unbiased):
  if unbiased:
    is_biased_str = "unbiased"
  else:
    is_biased_str = "biased"
  return is_biased_str + "-" + bias_name


def get_predictions(path, bias_name: str, unbiased: bool, bach_size=128, sample=None, n_processes=None, cache=False):
  dataset_name = get_dataset_name(bias_name, unbiased)

  output = join(path, "%s-prediction.pkl" % dataset_name)
  if sample is None and exists(output) and cache:
    return py_utils.load_pickle(output)

  logging.info("Computing predictions for %s on %s..." % (path, dataset_name))
  logging.info("Loading model...")
  model_dir = ModelDir(path)
  model = model_dir.get_model()

  logging.info("Setup data...")
  all_examples = mnli.load_mnli(False)
  all_examples = mnli.tokenize_examples(all_examples, model.get_tokenizer(), n_processes)
  all_examples = add_bias(all_examples, bias_name, unbiased)
  all_examples.sort(key=lambda x: len(x.premise))

  voc = set()
  for ex in all_examples:
    voc.update(ex.premise)
    voc.update(ex.hypothesis)

  model.set_vocab(voc)

  with tf.Session(graph=tf.Graph()) as sess:
    ds = mnli.make_dataset(all_examples, shuffle=False)
    fn = model.tensorize_fn()

    ds = ds.map(fn)
    ds = ds.padded_batch(bach_size, ds.output_shapes)
    ds.prefetch(5)
    it = ds.make_initializable_iterator()

    next_op = it.get_next()
    logits = model.apply(False, next_op, None)
    pred_op = tf.nn.softmax(logits)

    logging.info("Initializing...")
    if sess is None:
      sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(it.initializer)

    logging.info("Loading checkpoint...")
    saver = tf.train.Saver()
    saver.restore(sess, model_dir.get_latest_checkpoint())

    predictions = []
    pbar = tqdm(desc="classify", total=len(all_examples), ncols=80)
    while True:
      try:
        predictions.append(sess.run(pred_op))
        pbar.update(len(predictions[-1]))
      except tf.errors.OutOfRangeError:
        break
    pbar.close()

  predictions = np.concatenate(predictions, 0)
  predictions = {k.id: p for p, k in zip(predictions, all_examples)}
  if sample is None and cache:
    with open(output, "wb") as f:
      pickle.dump(predictions, f)
  return predictions


def compute_scores(path, bias_name, unbiased, cache=True, n_processes=None):
  dataset_name = get_dataset_name(bias_name, unbiased)
  cache_file = join(path, "%s-scores.json" % dataset_name)
  if exists(cache_file) and cache:
    return py_utils.load_json(cache_file)
  else:
    print("Scoring %s..." % path)
    pred = get_predictions(path, bias_name, unbiased, n_processes=n_processes, cache=cache)
    data = mnli.load_mnli(False)
    label_arr = np.array([x.label for x in data])
    pred_arr = np.array([pred[x.id] for x in data])
    pred_arr = np.argmax(pred_arr, 1)
    acc = (label_arr == pred_arr).mean()
    result = {"accuracy": acc}

    if cache:
      with open(cache_file, "w") as f:
        json.dump(result, f)
    return result


def show_scores(path, bias_name, is_biased: List[bool], cache=True, n_processes=None):
  result_dict = {}

  for b in is_biased:
    result_dict[b] = compute_scores(path, bias_name, b, cache, n_processes)

  for k, result in result_dict.items():
    print("*"*8 + " " + bias_name + " " + ("unbiased" if k else "biased") + " " + "*"*8)
    print(json.dumps(result, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("--n_processes", "-n", type=int, default=1)
  parser.add_argument("--bias", choices=["indicator", "excluder", 'depedent'],
                      required=True)
  parser.add_argument("--on", choices=["biased", "unbiased", 'both'], default="unbiased")
  parser.add_argument("--nocache", action="store_true")
  args = parser.parse_args()
  py_utils.add_stdout_logger()

  if args.on == "biased":
    is_biased = [True]
  elif args.on == "unbiased":
    is_biased = [False]
  elif args.on == "both":
    is_biased = [True, False]
  else:
    raise RuntimeError(args.on)

  show_scores(args.model, args.bias, is_biased, not args.nocache, args.n_processes)


if __name__ == '__main__':
  main()
