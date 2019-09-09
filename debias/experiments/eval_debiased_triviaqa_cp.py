import argparse
import json
import logging
from os.path import join, exists

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from debias.datasets import triviaqa_cp
from debias.models.model_dir import ModelDir
from debias.utils import py_utils, ops
from triviaqa_cp import triviaqa_cp_evaluation


def get_cache_name(dataset_name, part_name):
  return dataset_name + "-" + part_name


def get_predictions(path, dataset_name, part, bach_size=128, sample=None, cache=True):
  output_file = join(path, "%s-predictions.json" % get_cache_name(dataset_name, part))
  if sample is None and cache and exists(output_file):
    return py_utils.load_json(output_file)

  logging.info("Computing predictions for %s on %s..." % (path, dataset_name))
  logging.info("Loading model...")
  model_dir = ModelDir(path)
  model = model_dir.get_model()

  logging.info("Setup data...")
  data = triviaqa_cp.load_annotated_triviaqa_cp(dataset_name, part)
  data.sort(key=lambda x: len(x.tokens))
  voc = triviaqa_cp.compute_voc(data)
  model.set_vocab(voc)
  tuples = triviaqa_cp.convert_to_tuples(data)
  if sample is not None:
    np.random.shuffle(tuples)
    tuples = tuples[:sample]

  with tf.Session(graph=tf.Graph()) as sess:
    ds = triviaqa_cp.make_dataset(tuples)
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
    tokens = tup[2]
    predicted_text[tup[0]] = " ".join(tokens[s:e+1])

  if sample is None and cache:
    with open(output_file, "w") as f:
      json.dump(predicted_text, f)
  return predicted_text


def compute_scores(path, dataset_name, part, cache=True):
  output_file = join(path, "%s-scores.json" % get_cache_name(dataset_name, part))
  if cache and exists(output_file):
    return py_utils.load_json(output_file)

  logging.info("Scoring on %s" % dataset_name)
  docs = triviaqa_cp.load_triviaqa_cp(dataset_name, part)
  ground_truth = {x['QuestionId']: x['Answer'] for x in docs}

  pred = get_predictions(path, dataset_name, part, cache=cache)
  result = triviaqa_cp_evaluation.evaluate_triviaqa(ground_truth, pred, mute=True)

  if cache:
    with open(output_file, "w") as f:
      json.dump(result, f)
  return result


def show_scores(path, dataset, parts, cache=True):
  results = {}
  logging.info("Evaluating on %s" % dataset)
  for p in parts:
    results[dataset] = compute_scores(path, dataset, p, cache)

  print()
  for k, v in results.items():
    print("*"*8 + " TriviaQA-CP " + k + " Test " + "*"*8)
    print(json.dumps(v, indent=2))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("output_dir")
  parser.add_argument("--nocache", action="store_true")
  parser.add_argument("--dataset", choices=["location", "person"], required=True,
                      help="Dataset to test on")
  parser.add_argument("--parts", default=None, help="Comma seperated list of parts to test on")
  args = parser.parse_args()
  py_utils.add_stdout_logger()

  if args.parts is None:
    parts = ["dev", "test"]
  else:
    parts = args.parts.split(",")
    for ds in parts:
      if ds not in ["dev", "test", "train"]:
        raise ValueError("Unsupported dataset %s" % ds)

  show_scores(args.output_dir, args.dataset, parts, not args.nocache)


if __name__ == '__main__':
  main()
