import logging
import pickle
import socket
from datetime import datetime
from os import listdir
from os.path import join, exists
from shutil import rmtree
from typing import Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from debias.datasets.dataset_utils import QuantileBatcher
from debias.datasets.training_data_loader import TrainingDataLoader
from debias.models.model_dir import ModelDir
from debias.models.text_model import TextModel
from debias.training.evaluator import Evaluator
from debias.utils import configured
from debias.utils.configured import Configured


class AdamOptimizer(configured.Configured):
  def __init__(self, learning_rate=0.001, decay_steps=100, decay_rate=0.999,
               staircase=True, max_grad_norm=None):
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.decay_rate = decay_rate
    self.staircase = staircase
    self.decay_steps = decay_steps

  def get_train_op(self, loss, var_list=None):
    lr = self.learning_rate
    if self.decay_rate is not None:
      gs = tf.train.get_global_step()
      lr = tf.train.exponential_decay(
        global_step=gs, learning_rate=lr,
        staircase=self.staircase, decay_steps=self.decay_steps,
        decay_rate=self.decay_rate)

    opt = tf.train.AdamOptimizer(learning_rate=lr)

    grad_and_vars = opt.compute_gradients(loss, var_list=var_list)

    if self.max_grad_norm is not None:
      grads = [x[0] for x in grad_and_vars]
      grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
      grad_and_vars = [(g, v) for g, (_, v) in zip(grads, grad_and_vars)]

    return opt.apply_gradients(grad_and_vars, None)


class Trainer(Configured):

  def __init__(
      self, train_batcher: QuantileBatcher,
      optimizer: AdamOptimizer, evaluator: Evaluator,
      eval_batch_size,
      num_epochs: int, epoch_size: int,
      tensorize_par_calls=1, learning_rate=0.001, log_period: int=100,
      prefetch: int=5, max_checkpoints_to_keep: int=1,
      loss_ema: Optional[float] = 0.999, n_processes: int=1,
      progress_bar=True, seed: int=None
  ):
    self.optimizer = optimizer
    self.epoch_size = epoch_size
    self.evaluator = evaluator
    self.train_batcher = train_batcher
    self.eval_batch_size = eval_batch_size
    self.learning_rate = learning_rate
    self.seed = seed
    self.max_checkpoints_to_keep = max_checkpoints_to_keep
    self.loss_ema = loss_ema
    self.num_epochs = num_epochs
    self.log_period = log_period
    self.n_processes = n_processes
    self.progress_bar = progress_bar
    self.prefetch = prefetch
    self.tensorize_par_calls = tensorize_par_calls

  def train(
      self,
      data: TrainingDataLoader,
      model: TextModel,
      out: str,
      notes: str = None,
      config=None,
      sess=None
  ):
    if out is not None:
      if exists(out) and len(listdir(out)) > 0:
        if input("Files already exist in %s, override (y/n)?" % out).strip() == "y":
          rmtree(out)
        else:
          raise ValueError("Files already exist in %s" % out)
      out = ModelDir(out)

    tokenizer = model.get_tokenizer()
    logging.info("Loading data...")
    training_data = data.load(tokenizer, self.n_processes)

    logging.info("Setting up datasets...")
    model.set_vocab(training_data.voc)
    tensorize_fn = model.tensorize_fn()

    train = training_data.train.repeat()
    train = train.map(tensorize_fn, num_parallel_calls=self.tensorize_par_calls)
    train = self.train_batcher.batch(train)
    train = train.prefetch(self.prefetch)

    eval_datasets = training_data.eval_sets
    for k, ds in eval_datasets.items():
      ds = ds.map(tensorize_fn, num_parallel_calls=self.tensorize_par_calls)
      ds = ds.padded_batch(self.eval_batch_size, ds.output_shapes)
      ds = ds.prefetch(self.prefetch)
      eval_datasets[k] = ds

    if sess is None:
      logging.info("Initializing session...")
      sess = tf.Session(config=config)

    if self.seed is not None:
      tf.set_random_seed(self.seed)

    global_step = tf.get_variable('global_step', shape=(), dtype='int32',
                    initializer=tf.constant_initializer(0), trainable=False)
    add_global_step = tf.assign_add(global_step, tf.ones((), global_step.dtype))
    tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)

    logging.info("Building graph...")

    # **** train op ****
    train_it = train.make_initializable_iterator()
    train_input_op = train_it.get_next()

    with tf.name_scope("train"):
      model.apply(True, train_input_op, train_input_op["label"])

    losses = tf.get_collection(tf.GraphKeys.LOSSES, "train")
    if len(losses) == 0:
      raise ValueError("Model did not add any losses")
    loss = tf.add_n(losses)
    if len(loss.shape) != 0:
      raise ValueError("Loss is not a scalar, has shape %s" % loss.shape.as_list())

    train_op = self.optimizer.get_train_op(loss)

    train_update_ops = []

    # EMA for the loss
    if self.loss_ema is not None:
      loss_ema = tf.train.ExponentialMovingAverage(decay=self.loss_ema, name="LossEMA", zero_debias=True)
      train_update_ops.append(loss_ema.apply([loss]))
      report_loss_op = loss_ema.average(loss)
    else:
      report_loss_op = loss

    tf.summary.scalar("monitor/loss", report_loss_op)

    summary_op = tf.summary.merge_all()

    # Finally, merge the training op and any other op into one op
    # that additionally computes the loss
    train_update_ops.append(train_op)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
      with tf.control_dependencies(train_update_ops):
          train_op = tf.identity(loss)

    # **** build evaluation ops ****
    if self.evaluator is not None:
      self.evaluator.setup(eval_datasets, model)

    # Savers to record/log as we go
    if out is not None:
      saver = tf.train.Saver(max_to_keep=self.max_checkpoints_to_keep, save_relative_paths=True)
      summary_writer = tf.summary.FileWriter(out.log_dir)
    else:
      saver = None
      summary_writer = None

    logging.info("Initializing...")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(train_it.initializer)

    # Initialize the output dir
    # We do this last so if there are bugs in the setup nothing will have been written yet
    if out is not None:
      with open(join(out.dir, "model.json"), "w") as f:
        f.write(configured.config_to_json(model, indent=2))

      with open(join(out.dir, "data.json"), "w") as f:
        f.write(configured.config_to_json(data, indent=2))

      if notes is not None:
        with open(join(out.dir, "notes.txt"), "w") as f:
          f.write(notes)

      hostname = socket.gethostname()
      train = dict(
        trainer=self,
        evaluator=self.evaluator,
        date=datetime.now().strftime("%m%d-%H%M%S"),
        host=hostname
      )
      with open(join(out.dir, "trainer.json"), "w") as f:
        f.write(configured.config_to_json(train, indent=2))

      # Model also saved via pickle since I didn't want to deal with
      # doing the json->python conversion
      with open(join(out.dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    on_step = sess.run(global_step)

    # Make sure a bug doesn't cause us to add more ops later
    tf.get_default_graph().finalize()

    logging.info("Start training!")

    for epoch in range(1, self.num_epochs+1):
      if self.progress_bar:
        pbar = tqdm(total=self.epoch_size, desc="ep=%d" % epoch, ncols=100)
      else:
        pbar = None

      for _ in range(self.epoch_size):
        on_step = sess.run(global_step) + 1
        get_summary = on_step % self.log_period == 0

        if get_summary:
          summary, batch_loss, report_loss = sess.run([summary_op, train_op, report_loss_op])
        else:
          summary = None
          batch_loss, report_loss = sess.run([train_op, report_loss_op])

        if np.isnan(batch_loss):
          raise RuntimeError("NaN loss!")
        if np.isinf(batch_loss):
          raise RuntimeError("Infinity loss!")

        sess.run(add_global_step)

        if pbar is not None:
          pbar.update(1)
          descript = "ep=%d loss=%.4f" % (epoch, report_loss)
          pbar.set_description(descript, refresh=False)

        if summary is not None and out is not None:
          summary_writer.add_summary(summary, on_step)

      # Finished the training for this epoch, now save/evaluate
      if pbar is not None:
        pbar.close()

      if out is not None:
        saver.save(sess, join(out.save_dir, "checkpoint"), global_step=global_step)

      logging.info("Running evaluation %d..." % epoch)
      for eval_name in eval_datasets:
        results, summaries = self.evaluator.run(sess, eval_name)
        logging.info("%s: %s", eval_name, " ".join("%s=%.4f" % (k, v) for k, v in results.items()))
        if summary_writer is not None:
          for sum in summaries:
            summary_writer.add_summary(sum, on_step)

    summary_writer.close()

