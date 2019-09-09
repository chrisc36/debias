from os import mkdir
from os.path import join, exists

import tensorflow as tf

from debias.models.text_model import TextModel
from debias.utils.py_utils import load_pickle


class ModelDir(object):
  """ Wrapper for accessing a folder we are storing a model in"""

  def __init__(self, name: str):
    self.dir = name

  def get_model_file(self) -> str:
    return join(self.dir, "model.pkl")

  def get_model(self) -> TextModel:
    return load_pickle(join(self.dir, "model.pkl"))

  def get_eval_dir(self) -> str:
    answer_dir = join(self.dir, "answers")
    if not exists(answer_dir):
      mkdir(answer_dir)
    return answer_dir

  def get_latest_checkpoint(self):
    return tf.train.latest_checkpoint(self.save_dir)

  @property
  def save_dir(self):
    # Stores training checkpoint
    return join(self.dir, "save")

  @property
  def log_dir(self):
    return join(self.dir, "log")
