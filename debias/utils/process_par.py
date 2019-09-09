from multiprocessing import Lock
from multiprocessing import Pool
from typing import Iterable, List

from tqdm import tqdm

from debias.utils.py_utils import split, flatten_list, group


class Processor:

  def process(self, data: Iterable):
    """Map elements to an unspecified output type, the output but type must None or
    be able to be aggregated with the  `+` operator"""
    raise NotImplementedError()

  def finalize_chunk(self, data):
    """Finalize the output from `preprocess`, in multi-processing senarios this will still be run on
     the main thread so it can be used for things like interning"""
    pass


def _process_and_count(questions: List, preprocessor: Processor):
  count = len(questions)
  output = preprocessor.process(questions)
  return output, count


def process_par(data: List, processor: Processor, n_processes,
                chunk_size=1000, desc=None, initializer=None):
  """Runs `processor` on the elements in `data`, possibly in parallel, and monitor with tqdm"""

  if chunk_size <= 0:
    raise ValueError("Chunk size must be >= 0, but got %s" % chunk_size)
  if n_processes is not None and n_processes <= 0:
    raise ValueError("n_processes must be >= 1 or None, but got %s" % n_processes)
  n_processes = min(len(data), 1 if n_processes is None else n_processes)

  if n_processes == 1 and not initializer:
    out = processor.process(tqdm(data, desc=desc, ncols=80))
    processor.finalize_chunk(out)
    return out
  else:
    chunks = split(data, n_processes)
    chunks = flatten_list([group(c, chunk_size) for c in chunks])
    total = len(data)
    pbar = tqdm(total=total, desc=desc, ncols=80)
    lock = Lock()

    def call_back(results):
      processor.finalize_chunk(results[0])
      with lock:
        pbar.update(results[1])

    with Pool(n_processes, initializer=initializer) as pool:
      results = [
        pool.apply_async(_process_and_count, [c, processor], callback=call_back)
        for c in chunks
      ]
      results = [r.get()[0] for r in results]

    pbar.close()
    output = results[0]
    if output is not None:
      for r in results[1:]:
        output += r
    return output
