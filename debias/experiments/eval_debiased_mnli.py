import argparse
import json
import logging
import pickle
from os.path import join, exists

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from debias.datasets import mnli
from debias.models.model_dir import ModelDir
from debias.utils import py_utils


def load_eval_set(dataset_name, sample=None):
  if dataset_name == "hans":
    return mnli.load_hans(sample)
  elif dataset_name == "dev":
    return mnli.load_mnli(False, sample)
  else:
    raise NotImplementedError()


def get_predictions(path, part="hans", bach_size=128, sample=None, n_processes=None, cache=False):
  output = join(path, "%s-prediction.pkl" % part)
  if sample is None and exists(output) and cache:
    return py_utils.load_pickle(output)

  logging.info("Computing predictions for %s on %s..." % (path, part))
  logging.info("Loading model...")
  model_dir = ModelDir(path)
  model = model_dir.get_model()
  tokenizer = model.get_tokenizer()
  all_examples = load_eval_set(part, sample)
  all_examples = mnli.tokenize_examples(all_examples, tokenizer, n_processes)

  all_examples.sort(key=lambda x: len(x.premise))

  logging.info("Setup data...")
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


def compute_dev_score(predictions, dev_examples):
  pred = np.stack([predictions[x.id] for x in dev_examples], 0)
  label_arr = np.array([x.label for x in dev_examples])
  return (np.argmax(pred, 1) == label_arr).mean()


def compute_hans_score(preds, hans_examples, mode="sum"):
  label_arr = np.array([x.label for x in hans_examples])
  pred = np.stack([preds[x.id] for x in hans_examples], 0)
  if mode == "sum":
    pred = np.argmax(np.stack([
      pred[:, 0] + pred[:, 2],
      pred[:, 1]
    ], 1), 1)
  elif mode == "map":
    pred = np.argmax(pred, 1)
    pred[pred == 2] = 0
  else:
    raise NotImplementedError(mode)

  correct = pred == label_arr
  return correct.mean()


def get_hans_scores(path, cache=True, n_processes=None, mode="sum"):
  cache_file = join(path, "hans_scores.json")
  if exists(cache_file) and cache and mode == "sum":
    return py_utils.load_json(cache_file)
  else:
    logging.info("Scoring %s..." % path)
    pred = get_predictions(path, "hans", n_processes=n_processes, cache=cache)
    hans = load_eval_set("hans")

    score = compute_hans_score(pred, hans, mode)
    score = {"accuracy": score}
    if cache and mode == "sum":
      with open(cache_file, "w") as f:
        json.dump(score, f)
    return score


def get_dev_scores(path, cache=True, n_processes=None):
  cache_file = join(path, "dev_scores.json")
  if exists(cache_file) and cache:
    return py_utils.load_json(cache_file)
  else:
    logging.info("Scoring %s..." % path)
    pred = get_predictions(path, "dev", n_processes=n_processes, cache=cache)
    hans = load_eval_set("dev")
    score = compute_dev_score(pred, hans)
    score = {"accuracy": score}
    if cache:
      with open(cache_file, "w") as f:
        json.dump(score, f)
    return score


def show_scores(path, dev: bool, hans: bool, cache=True, n_processes=None):
  result_dict = {}
  if dev:
    result_dict["dev"] = get_dev_scores(path, n_processes=n_processes, cache=cache)
  if hans:
    result_dict["hans"] = get_hans_scores(path, n_processes=n_processes, cache=cache)
  for k, result in result_dict.items():
    print("*"*8 + " " + k + " " + "*"*8)
    print(json.dumps(result, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model")
  parser.add_argument("--n_processes", "-n", type=int, default=1)
  parser.add_argument("--nocache", action="store_true")
  parser.add_argument("--dataset", choices=["dev", "hans", "both"],
                      default="both")
  args = parser.parse_args()
  py_utils.add_stdout_logger()

  compute_scores(
    args.model, args.dataset in ["dev", "both"], args.dataset in ["hans", "both"],
    not args.nocache, args.n_processes)


if __name__ == '__main__':
  main()
