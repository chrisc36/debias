import argparse
import pickle
from os import mkdir
from os.path import join, exists

from tensorflow.python.framework.errors_impl import OutOfRangeError
from tqdm import tqdm

from debias.datasets.dataset_utils import QuantileBatcher, build_epoch_fn
from debias.datasets.squad import AnnotatedSquadLoader, load_annotated_squad
from debias.datasets.training_data_loader import TrainingDataLoader, TrainingData, PREMISE_KEY, \
  PREMISE_LEN_KEY
from debias.models.model_dir import ModelDir
from debias.models.text_model import TextModel
from debias.modules.qa_debias_loss_functions import compute_nll
from debias.preprocessing.squad_tfidf_features import get_squad_tfidf_features
from debias.training.trainer import Trainer, AdamOptimizer
from debias.utils import ops
from debias.utils.configured import Configured
from debias.utils.tokenizer import Tokenizer, NltkAndPunctTokenizer
import tensorflow as tf
import numpy as np


def elementwise_logsumexp(x, y):
  return tf.maximum(x, y) + tf.log1p(tf.exp(-tf.abs(x - y)))


class ScalarFeaturePredictor(Configured):
  """A simple model that predicts the target span using a scalar per-word feature"""

  def __init__(self, plain=True, log=None, sumexp=False):
    self.plain = plain
    self.log = log
    self.sumexp = sumexp

  def get_tokenizer(self):
    return None

  def set_vocab(self, _):
    pass

  def tensorize_fn(self):
    def fn(x):
      return x

    return fn

  def apply(self, is_train, features, labels):
    p_mask = features[PREMISE_LEN_KEY]
    feature = features[PREMISE_KEY]  # The feature to use during prediction

    fe = []  # Transformation of the feature to predict with
    if self.plain:
      fe.append(feature)
    if self.log:
      if isinstance(self.log, list):
        for l in self.log:
          fe.append(tf.log(feature + l))
      else:
        fe.append(tf.log(feature + self.log))
    if self.sumexp:
      fe.append(elementwise_logsumexp(tf.log(feature + 0.1), tf.get_variable("offset", ())))
    feature = tf.stack(fe, 2)

    start_scores = ops.last_dim_weighted_sum(feature, "start-w")
    start_scores = ops.mask_logits(start_scores, p_mask)
    end_scores = ops.last_dim_weighted_sum(feature, "end-w")
    end_scores = ops.mask_logits(end_scores, p_mask)
    logits = tf.stack([start_scores, end_scores], -1)

    if labels is not None:
      if len(labels.shape.as_list()) == 2:
        # Sparse label of size [batch, 2] of target start/end tokens
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels[:, 0],
          logits=start_scores
        ))
        loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels[:, 1],
          logits=end_scores
        ))
      else:
        # Dense label of size [batch, seq_len, 2] with true/false for target start/end tokens
        loss = tf.reduce_mean(compute_nll(logits, labels, None))
      tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

    return tf.nn.log_softmax(logits, 1)


def make_dataset(features, sample=None, shuffle=False):
  get = build_epoch_fn(features, sample, shuffle=shuffle)
  if len(features[0][2].shape) == 2:
    label_shape = (None, 2)
    label_dtype = tf.bool
  else:
    label_shape = (2,)
    label_dtype = tf.int32

  data = tf.data.Dataset.from_generator(
    get, (tf.string, tf.float32, label_dtype), ((), (None,), label_shape))

  def to_map(qid, features, labels):
    return {
      "qid": qid,
      PREMISE_KEY: features,
      "label": labels,
      PREMISE_LEN_KEY: tf.shape(features)[0]
    }

  return data.map(to_map)


def load_squad_with_features(key, pos_filtered, multi_label):
  fe = get_squad_tfidf_features(key, pos_filtered)
  data = load_annotated_squad(key)
  out = []
  for ex in data:
    for q in ex.questions:
      if multi_label:
        dense_answers = np.zeros((len(ex.tokens), 2), np.bool)
        for s, e in q.answer_spans:
          dense_answers[s, 0] = True
          dense_answers[e, 1] = True

        out.append((q.question_id, fe[q.question_id], dense_answers))
      else:
        out.append((q.question_id, fe[q.question_id], q.answer_spans[0]))

  out.sort(key=lambda x: -len(x[0]))
  return out


class SquadTfidfFeaturesLoader(TrainingDataLoader):

  def __init__(self, pos_filtered, train_sample, multi_label):
    self.pos_filtered = pos_filtered
    self.train_sample = train_sample
    self.multi_label = multi_label

  def load(self, tokenizer, n_processes=None):
    train = load_squad_with_features("train", self.pos_filtered, self.multi_label)
    train_ds = make_dataset(train, None, shuffle=True)
    return TrainingData(train_ds, {}, None)


def build_squad_bias_only_model(model_dir, output_dir, pos_filtered):
  """Train the SQuAD Bias-Only model

  :param model_dir: Directory to save the model
  :param output_dir: Director to save the model predictions
  :param pos_filtered: Should we use pos-filtered TF-IDF features or not
  """
  debias / preprocessing / train_squad_bias.py
  opt = AdamOptimizer(
    learning_rate=0.001,
    max_grad_norm=5.0,
    staircase=True,
    decay_steps=100,
    decay_rate=0.999
  )

  data_loader = SquadTfidfFeaturesLoader(pos_filtered, 5000, False)

  tr = Trainer(
    QuantileBatcher(45, 10, 300, 4, 12),
    opt,
    evaluator=None,
    eval_batch_size=90,
    epoch_size=1341, num_epochs=10, log_period=100,
  )

  model = ScalarFeaturePredictor(sumexp=True, log=0.1)

  print("Training...")
  with tf.Session(graph=tf.Graph()) as sess:
    tr.train(data_loader, model, model_dir, sess=sess)

  print("Evaluating...")
  train = load_squad_with_features("train", pos_filtered, data_loader.multi_label)
  total = len(train)
  train = make_dataset(train)
  train = train.padded_batch(45, train.output_shapes).prefetch(5)

  eval_it = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
  eval_input_op = eval_it.get_next()
  logit_op = model.apply(False, eval_input_op, None)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  sess.run(tf.tables_initializer())
  saver = tf.train.Saver()
  saver.restore(sess, ModelDir(model_dir).get_latest_checkpoint())

  if not exists(output_dir):
    mkdir(output_dir)

  for dataset_name in ["train", "add_sent", "dev", "add_one_sent"]:
    out = {}
    if dataset_name == "train":
      sess.run(eval_it.make_initializer(train))
    else:
      ds = load_squad_with_features(dataset_name, pos_filtered, data_loader.multi_label)
      total = len(ds)
      ds = make_dataset(ds)
      ds = ds.padded_batch(45, ds.output_shapes).prefetch(5)
      sess.run(eval_it.make_initializer(ds))

    pbar = tqdm(desc=dataset_name, ncols=100, total=total)
    while True:
      try:
        logits, qids, p_lens = sess.run([logit_op, eval_input_op["qid"], eval_input_op[PREMISE_LEN_KEY]])
      except OutOfRangeError:
        break
      pbar.update(len(logits))
      for logit, qid, p_len in zip(logits, qids, p_lens):
        out[qid.decode("utf-8")] = logit[:p_len]
    pbar.close()

    with open(join(output_dir, dataset_name + ".pkl"), "wb") as f:
      pickle.dump(out, f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_dir", help="Directory to save the trained model")
  parser.add_argument("output_dir", help="Directory to save the model predictions")
  parser.add_argument("--pos_filtered", help="Use POS Filtered TF-IDF features",
                      action="store_true")
  args = parser.parse_args()
  build_squad_bias_only_model(args.model_dir, args.output_dir, args.pos_filtered)


if __name__ == '__main__':
  main()
