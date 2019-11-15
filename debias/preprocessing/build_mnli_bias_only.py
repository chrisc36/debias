"""Script to build the bias-only model for MNLI

Note running this also requires install pandas and sklearn
"""

import argparse
import logging
import pickle
from os import mkdir
from os.path import exists, join

from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import tensorflow as tf

from debias.datasets.mnli import load_hans, load_mnli, tokenize_examples
from debias.utils import py_utils
from debias.utils.load_word_vectors import load_word_vectors
from debias.utils.tokenizer import NltkAndPunctTokenizer


STOP_WORDS = frozenset([
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
  'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
  'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
  'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
  'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
  'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
  'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
  'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
  'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
  've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
  'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn',
   "many", "how", "de"
])


def is_subseq(needle, haystack):
  l = len(needle)
  if l > len(haystack):
    return False
  else:
    return any(haystack[i:i+l] == needle for i in range(len(haystack)-l + 1))


def build_mnli_bias_only(out_dir, cache_examples=None, w2v_cache=None):
  """Builds our bias-only MNLI model and saves its predictions

  :param out_dir: Directory to save the predictions
  :param cache_examples: Cache examples to this file
  :param w2v_cache: Cache w2v features to this file
  """
  py_utils.add_stdout_logger()

  tok = NltkAndPunctTokenizer()

  # Load the data we want to use
  if cache_examples and exists(cache_examples):
    tf.logging.info("Loading cached examples")
    with open(cache_examples, "rb") as f:
      dataset_to_examples = pickle.load(f)
  else:
    dataset_to_examples = {}
    dataset_to_examples["hans"] = tokenize_examples(load_hans(), tok, 5)
    dataset_to_examples["train"] = tokenize_examples(load_mnli(True), tok, 5)
    dataset_to_examples["dev"] = tokenize_examples(load_mnli(False), tok, 5)
    if cache_examples:
      with open(cache_examples, "wb") as f:
        pickle.dump(dataset_to_examples, f)

  # Our models will only distinguish entailment vs (neutral/contradict)
  for examples in dataset_to_examples.values():
    for i, ex in enumerate(examples):
      if ex.label == 2:
        examples[i] = ex._replace(label=0)

  # Load the pre-normalized word vectors to use when building features
  if w2v_cache and exists(w2v_cache):
    tf.logging.info("Loading cached word vectors")
    with open(w2v_cache, "rb") as f:
      w2v = pickle.load(f)
  else:
    logging.info("Loading word vectors")
    voc = set()
    for v in dataset_to_examples.values():
      for ex in v:
        voc.update(ex.hypothesis)
        voc.update(ex.premise)
    words, vecs = load_word_vectors("crawl-300d-2M", voc)
    w2v = {w: v/np.linalg.norm(v) for w, v in zip(words, vecs)}
    if w2v_cache:
      with open(w2v_cache, "wb") as f:
        pickle.dump(w2v, f)

  # Build the features, store as a pandas dataset
  dataset_to_features = {}
  for name, examples in dataset_to_examples.items():
    tf.logging.info("Building features for %s.." % name)
    features = []
    for example in examples:
      h = [x.lower() for x in example.hypothesis]
      p = [x.lower() for x in example.premise]
      p_words = set(p)
      n_words_in_p = sum(x in p_words for x in h)
      fe = {
        "h-is-subseq": is_subseq(h, p),
        "all-in-p": n_words_in_p == len(h),
        "percent-in-p": n_words_in_p / len(h),
        "log-len-diff": np.log(max(len(p) - len(h), 1)),
        "label": example.label
      }

      h_vecs = [w2v[w] for w in example.hypothesis if w in w2v]
      p_vecs = [w2v[w] for w in example.premise if w in w2v]
      if len(h_vecs) > 0 and len(p_vecs) > 0:
        h_vecs = np.stack(h_vecs, 0)
        p_vecs = np.stack(p_vecs, 0)
        # [h_size, p_size]
        similarities = np.matmul(h_vecs, p_vecs.T)
        # [h_size]
        similarities = np.max(similarities, 1)
        similarities.sort()
        fe["average-sim"] = similarities.sum() / len(h)
        fe["min-similarity"] = similarities[0]
        if len(similarities) > 1:
          fe["min2-similarity"] = similarities[1]

      features.append(fe)

    dataset_to_features[name] = pd.DataFrame(features)
    dataset_to_features[name].fillna(0.0, inplace=True)

  # Train the model
  tf.logging.info("Fitting...")
  train_df = dataset_to_features["train"]
  feature_cols = [x for x in train_df.columns if x != "label"]

  # class_weight='balanced' will weight the entailemnt/non-entailment examples equally
  # C=100 means no regularization
  lr = LogisticRegression(multi_class="auto", solver="liblinear",
                          class_weight='balanced', C=100)
  lr.fit(train_df[feature_cols].values, train_df.label.values)

  # Save the model predictions
  if not exists(out_dir):
    mkdir(out_dir)

  for name, ds in dataset_to_features.items():
    tf.logging.info("Predicting for %s" % name)
    examples = dataset_to_examples[name]
    pred = lr.predict_log_proba(ds[feature_cols].values).astype(np.float32)
    y = ds.label.values

    bias = {}
    for i in range(len(pred)):
      if examples[i].id in bias:
        raise RuntimeError("non-unique IDs?")
      bias[examples[i].id] = pred[i]

    acc = np.mean(y == np.argmax(pred, 1))
    print("%s two-class accuracy: %.4f (size=%d)" % (name, acc, len(examples)))

    with open(join(out_dir, "%s.pkl" % name), "wb") as f:
      pickle.dump(bias, f)


def main():
  parser = argparse.ArgumentParser("Train our MNLI bias-only model")
  parser.add_argument("output_dir", help="Directory to store the bias-only predictions")
  parser.add_argument("--cache_examples")
  parser.add_argument("--cache_w2v_features")
  args = parser.parse_args()

  build_mnli_bias_only(args.output_dir, args.cache_examples, args.cache_w2v_features)


if __name__ == "__main__":
  main()
