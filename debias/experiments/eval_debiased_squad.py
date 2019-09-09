import argparse
import json
import logging
from os.path import join, exists

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from debias.datasets import squad
from debias.models.model_dir import ModelDir
from debias.squad_eval import squad_v1_adversarial_evaluation, squad_v1_official_evaluation
from debias.utils import py_utils, ops


def get_predictions(path, dataset, bach_size=128, sample=None, cache=True):
  output_file = join(path, "%s-predictions.json" % dataset)
  if sample is None and cache and exists(output_file):
    return py_utils.load_json(output_file)

  logging.info("Computing predictions for %s on %s..." % (path, dataset))
  logging.info("Loading model...")
  model_dir = ModelDir(path)
  model = model_dir.get_model()

  logging.info("Setup data...")
  data = squad.load_annotated_squad(dataset)
  data.sort(key=lambda x: len(x.tokens))
  voc = squad.compute_voc(data)
  model.set_vocab(voc)
  tuples = squad.convert_to_tuples(data)
  if sample is not None:
    np.random.shuffle(tuples)
    tuples = tuples[:sample]

  with tf.Session(graph=tf.Graph()) as sess:
    ds = squad.make_dataset(tuples)
    fn = model.tensorize_fn()

    ds = ds.map(fn)
    ds = ds.padded_batch(bach_size, ds.output_shapes)
    ds.prefetch(5)
    it = ds.make_initializable_iterator()

    next_op = it.get_next()
    logit_op = model.apply(False, next_op, None)
    logit_op = tf.nn.log_softmax(logit_op, 1)
    span_op = ops.get_best_span(logit_op, 17)

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
    pbar = tqdm(desc="classify", total=len(tuples), ncols=80)
    while True:
      try:
        predictions.append(sess.run(span_op))
        pbar.update(len(predictions[-1]))
      except tf.errors.OutOfRangeError:
        break
    pbar.close()

  predictions = np.concatenate(predictions, 0)
  predicted_text = {}
  for tup, (s, e) in zip(tuples, predictions):
    passage_str, offsets, _, answer_text = tup[-4:]
    predicted_text[tup[0]] = (passage_str[offsets[s][0]:offsets[e][1]])

  if sample is None and cache:
    with open(output_file, "w") as f:
      json.dump(predicted_text, f)
  return predicted_text


def compute_scores(path, dataset_name, cache=True):
  output_file = join(path, "%s-scores.json" % dataset_name)
  if cache and exists(output_file):
    return py_utils.load_json(output_file)

  logging.info("Scoring on %s" % dataset_name)
  docs = squad.load_squad_documents(dataset_name)
  pred = get_predictions(path, dataset_name, cache=cache)
  if dataset_name in ["dev", "train"]:
    result = squad_v1_official_evaluation.evaluate(docs, pred)
  else:
    result = squad_v1_adversarial_evaluation.evaluate_adversarial(docs, pred)

  if cache:
    with open(output_file, "w") as f:
      json.dump(result, f)
  return result


def compute_all_scores(path, datasets, cache=True):
  results = {}
  for ds in datasets:
    logging.info("Evaluating on %s" % ds)
    results[ds] = compute_scores(path, ds, cache)

  print()
  for k, v in results.items():
    print("*" * 8 + " " + k + " " + "*"*8)
    print(json.dumps(v, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("output_dir")
  parser.add_argument("--nocache", action="store_true")
  parser.add_argument("--datasets", default=None, help="Comma separated list of datasets")
  args = parser.parse_args()
  py_utils.add_stdout_logger()

  if args.datasets is None:
    datasets = ["dev", "add_sent", "add_one_sent"]
  else:
    datasets = args.datasets.split(",")
    for ds in datasets:
      if ds not in squad.DATASETS:
        raise ValueError("Unsupported dataset %s" % ds)

  compute_all_scores(args.output_dir, datasets, not args.nocache)
  

if __name__ == '__main__':
  main()
