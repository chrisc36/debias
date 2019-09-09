import json
import logging
import math
import pickle
import sys
import tempfile
import zipfile
from os import makedirs
from os.path import dirname
from typing import List, Iterable, TypeVar

import numpy as np
import requests

# Try to avoid requiring tensorflow be installed for the utils methods
try:
  from tensorflow.python.util import deprecation as tf_deprecation
except ImportError:
  tf_deprecation = None

from tqdm import tqdm

T = TypeVar('T')


def load_pickle(filename):
  """Load an object from a pickled file."""
  with open(filename, "rb") as f:
    return pickle.load(f)


def load_json(filename):
  """Load an object from a json file."""
  with open(filename, "r") as f:
    return json.load(f)


def transpose_lists(lsts: Iterable[Iterable[T]]) -> List[List[T]]:
  """Transpose a list of lists."""
  return [list(i) for i in zip(*lsts)]


def flatten_list(iterable_of_lists: Iterable[Iterable[T]]) -> List[T]:
  """Unpack lists into a single list."""
  return [x for sublist in iterable_of_lists for x in sublist]


def split(lst: List[T], n_groups) -> List[List[T]]:
  """ partition `lst` into `n_groups` that are as evenly sized as possible  """
  per_group = len(lst) // n_groups
  remainder = len(lst) % n_groups
  groups = []
  ix = 0
  for _ in range(n_groups):
    group_size = per_group
    if remainder > 0:
      remainder -= 1
      group_size += 1
    groups.append(lst[ix:ix + group_size])
    ix += group_size
  return groups


def group(lst: List[T], max_group_size) -> List[List[T]]:
  """partition `lst` into that the mininal number of groups that as evenly sized
  as possible  and are at most `max_group_size` in size """
  if max_group_size is None:
    return [lst]
  n_groups = (len(lst) + max_group_size - 1) // max_group_size
  per_group = len(lst) // n_groups
  remainder = len(lst) % n_groups
  groups = []
  ix = 0
  for _ in range(n_groups):
    group_size = per_group
    if remainder > 0:
      remainder -= 1
      group_size += 1
    groups.append(lst[ix:ix + group_size])
    ix += group_size
  return groups


def add_stdout_logger():
  """Setup stdout logging"""
  if tf_deprecation is not None:
    # Tensorflow really wants to let us know about all the to-be-deprecated tf 1.13.1 functions,
    # some of which are called within the tensorflow library. Tell it to quite down
    tf_deprecation._PRINT_DEPRECATION_WARNINGS = False

  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S', )
  handler.setFormatter(formatter)
  handler.setLevel(logging.INFO)

  root = logging.getLogger()
  root.setLevel(logging.INFO)
  root.addHandler(handler)


def get_containing_spans(spans: np.ndarray, start: int, stop: int):
  """Get indices of the sorted spans in `spans` that overlap with `start` and `stop`"""
  idxs = []
  for word_ix, (s, e) in enumerate(spans):
    if e > start:
      if s < stop:
        idxs.append(word_ix)
      else:
        break
  return idxs


def ensure_dir_exists(filename):
  """Make sure the parent directory of `filename` exists"""
  makedirs(dirname(filename), exist_ok=True)


def download_to_file(url, output_file):
  """Download `url` to `output_file`, intended for small files."""
  ensure_dir_exists(output_file)
  with requests.get(url) as r:
    r.raise_for_status()
    with open(output_file, 'wb') as f:
      f.write(r.content)


def download_zip(name, url, source, progress_bar=True):
  """Download zip file at `url` and extract to `source`"""
  makedirs(source, exist_ok=True)
  logging.info("Downloading %s" % name)

  # Probably best to download to a temp file to ensure we
  # don't eat a lot of RAM with downloading a large file
  tmp_f = tempfile.TemporaryFile()
  with requests.get(url, stream=True) as r:
    _write_to_stream(r, tmp_f, progress_bar)

  logging.info("Extracting to %s...." % source)
  with zipfile.ZipFile(tmp_f) as f:
    f.extractall(source)


DRIVE_URL = "https://docs.google.com/uc?export=download"


def download_from_drive(file_id, output_file, progress_bar=False):
  """Download the public google drive file `file_id` to `output_file`"""
  ensure_dir_exists(output_file)

  session = requests.Session()

  response = session.get(DRIVE_URL, params={'id': file_id}, stream=True)

  # Check to see if we need to send a second, confirm, request
  # https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      params = {'id': file_id, 'confirm': value}
      response = session.get(DRIVE_URL, params=params, stream=True)

  with open(output_file, "wb") as f:
    _write_to_stream(response, f, progress_bar)
  response.close()


def _write_to_stream(response, output_fh, progress_bar=True, chunk_size=32768):
  """Write streaming `response` to `output_fs` in chunks"""
  mb = 1024*1024
  response.raise_for_status()
  if progress_bar:
    # tqdm does not format decimal numbers. We could in theory add decimal formatting
    # using the `bar_format` arg, but in practice doing so is finicky, in particular it
    # seems impossible to properly format the `rate` parameter. Instead we just manually
    # ensure the 'total' and 'n' values of the bar are rounded to the 10th decimal place
    content_len = response.headers.get("Content-Length")
    if content_len is not None:
      total = math.ceil(10 * float(content_len) / mb) / 10
    else:
      total = None
    pbar = tqdm(desc="downloading", total=total, ncols=100, unit="mb")
  else:
    pbar = None

  cur_total = 0
  for chunk in response.iter_content(chunk_size=chunk_size):
    if chunk:  # filter out keep-alive new chunks
      if pbar is not None:
        cur_total += len(chunk)
        next_value = math.floor(10 * cur_total / mb) / 10.0
        pbar.update(next_value - pbar.n)
      output_fh.write(chunk)

  if pbar is not None:
    if pbar.total is not None:
      pbar.update(pbar.total - pbar.n)  # Fix rounding errors just for neatness
    pbar.close()

