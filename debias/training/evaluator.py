from typing import Dict

import tensorflow as tf
from tqdm import tqdm

from debias.datasets.training_data_loader import PREMISE_LEN_KEY
from debias.models.text_model import TextModel
from debias.squad_eval.squad_eval import eval_squad_op
from debias.squad_eval.triviaqa_eval import eval_triviaqa_op
from debias.utils import configured


class Evaluator(configured.Configured):
  """Can be used to evaluate a model against multiple datasets multiple times"""

  def __init__(self, mode, progress_bar=True):
    self.progress_bar = progress_bar
    self.mode = mode

    self.eval_dataset_iterators = None
    self.eval_results = None
    self.eval_update_ops = None
    self.eval_summaries = None
    self.init_eval_vars = None
    self.eval_batch_size = None
    self.eval_max_sizes = None
    self.task_name = None

  def setup(self, eval_datasets: Dict[str, tf.data.Dataset], model: TextModel, reuse=True):
    """Build ops needed to evaluate on each (already batched) dataset in `eval_datasets`"""

    name = "eval"
    with tf.name_scope(name):
      with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        example = list(eval_datasets.values())[0]
        eval_it = tf.data.Iterator.from_structure(example.output_types, example.output_shapes)
        eval_op = eval_it.get_next()
        eval_pred = model.apply(False, eval_op, eval_op["label"])

    with tf.variable_scope(name):
      if self.mode == "clf":
        eval_pred = tf.nn.log_softmax(eval_pred)
        op, up = tf.metrics.accuracy(eval_op["label"], tf.argmax(eval_pred, 1))
        self.eval_update_ops = [up]
        self.eval_results = {"accuracy": op}
      elif self.mode == "squad":
        label = eval_op["label"]
        eval_pred = tf.nn.log_softmax(eval_pred, 1)
        scores = eval_squad_op(eval_pred, label["token_offsets"], label["passage_str"], label["answers"], 17)
        em_op, em_up = tf.metrics.mean(scores[:, 0])
        f1_op, f1_up = tf.metrics.mean(scores[:, 1])
        self.eval_update_ops = [em_up, f1_up]
        self.eval_results = {"em": em_op, "f1": f1_op}
      elif self.mode == "triviaqa":
        label = eval_op["label"]
        eval_pred = tf.nn.log_softmax(eval_pred, 1)
        scores = eval_triviaqa_op(eval_pred, eval_op["premise_tok"], label["answers"], 8)
        em_op, em_up = tf.metrics.mean(scores[:, 0])
        f1_op, f1_up = tf.metrics.mean(scores[:, 1])
        self.eval_update_ops = [em_up, f1_up]
        self.eval_results = {"em": em_op, "f1": f1_op}
      else:
        raise NotImplementedError(self.mode)

    self.eval_dataset_iterators = {k: eval_it.make_initializer(d) for k, d in eval_datasets.items()}
    self.eval_summaries = {}
    for dataset_name in eval_datasets:
      self.eval_summaries[dataset_name] = [tf.summary.scalar(dataset_name + "/" + k, v) for
                                           k, v in self.eval_results.items()]

    self.init_eval_vars = tf.variables_initializer(tf.local_variables(name))
    self.eval_batch_size = tf.shape(eval_op[PREMISE_LEN_KEY])[0]
    self.eval_max_sizes = {}

  def run(self, sess: tf.Session, eval_name: str, return_summaries=True):
    """Run evaluation on `eval_name`, which should refer to a dataset passed into `self.setup`"""

    sess.run([self.eval_dataset_iterators[eval_name], self.init_eval_vars])

    estimate_steps = self.eval_max_sizes.get(eval_name)
    max_steps = None

    if self.progress_bar:
      pbar = tqdm(desc=eval_name, ncols=80, total=estimate_steps)
    else:
      pbar = None
    total_size = 0

    up = self.eval_update_ops
    r = self.eval_results

    while True:
      try:
        bs, _ = sess.run([self.eval_batch_size, up])
        total_size += bs
        if max_steps is not None and total_size >= max_steps:
          total_size -= bs
          if pbar is not None:
            pbar.update(max_steps - total_size)
          break
        else:
          if pbar is not None:
            pbar.update(bs)
      except tf.errors.OutOfRangeError:
        break
    if pbar is not None:
      pbar.close()
    self.eval_max_sizes[eval_name] = max(total_size, self.eval_max_sizes.get(eval_name, 0))

    if return_summaries:
      return sess.run([r, self.eval_summaries[eval_name]])
    else:
      return sess.run(r)
