"""Preprocess the SQuAD dataset with CoreNLP tokenization and tagging"""
import argparse
import json
import pickle
import regex
from os import mkdir, makedirs
from os.path import join, exists, dirname
from typing import Iterable, List, Dict

import numpy as np

from debias import config
from debias.datasets.squad import AnnotatedSquadParagraph, SquadQuestion
from debias.preprocessing.corenlp_client import CoreNLPClient
from debias.utils import py_utils
from debias.utils.process_par import Processor, process_par
from debias.utils.py_utils import get_containing_spans


class SquadAnnotator(Processor):
  """Builds `AnnotatedSquadParagraph` objects from the SQuAD paragraphs as loaded from JSON."""

  def __init__(self, port, intern=False, resplit=True):
    self.port = port
    self.intern = intern
    self.resplit = resplit
    if self.resplit:
      resplit = r"\p{Pd}\p{Po}\p{Ps}\p{Pe}\p{S}\p{Pc}"
      resplit = "([" + resplit + "]|'')"
      split_regex = r"(?![\.,'])" + resplit
      self.split_regex = regex.compile(split_regex)

  def process(self, data: Iterable[Dict]) -> List[AnnotatedSquadParagraph]:
    client = CoreNLPClient(port=self.port)
    out = []

    for para in data:
      passage = para['context']

      offset = 0
      while passage[offset].isspace():
        offset += 1
      annotations = client.query_ner(passage[offset:])["sentences"]

      if self.resplit:
        # We re-split the CORENLP tokens on some punctuation tags, since we need pretty aggressive tokenization
        # in ensure (almost) all answers span are contained within tokens
        words, pos, ner, inv = [], [], [], []
        sentence_lens = []
        on_len = 0
        for sentences in annotations:
          if len(sentences["tokens"]) == 0:
            raise RuntimeError()
          for token in sentences["tokens"]:
            p, n = token["pos"], token["ner"]
            s, e = (token["characterOffsetBegin"], token["characterOffsetEnd"])
            if len(token["originalText"]) != (e - s):
              # For some reason (probably due to unicode-shenanigans) the character offsets
              # we get make are sometime incorrect, we fix it here
              offset -= (e - s) - len(token["originalText"])
            s += offset
            e += offset

            w = passage[s:e]

            if w == "''" or w == '``':
              split = [w]
            else:
              split = [x for x in self.split_regex.split(w) if len(x) > 0]

            if len(split) == 1:
              words.append(w)
              pos.append(p)
              ner.append(n)
              inv.append((s, e))
            else:
              words += split
              ner += [n] * len(split)
              pos += ['SEP' if self.split_regex.match(x) else p for x in split]

              for w in split:
                inv.append((s, s + len(w)))
                s += len(w)
              if s != e:
                raise RuntimeError()

          sentence_lens.append(len(words) - on_len)
          on_len = len(words)
      else:
        raise NotImplementedError()

      inv = np.array(inv, np.int32)
      sentence_lens = np.array(sentence_lens, np.int32)
      if sum(sentence_lens) != len(words):
        raise RuntimeError()

      questions = []
      for question in para["qas"]:
        q_tokens = py_utils.flatten_list([x["tokens"] for x in client.query_tokenize(question["question"])["sentences"]])

        answer_spans = []
        answers_text = []
        for answer_ix, answer in enumerate(question['answers']):
          answer_raw = answer['text']
          answer_start = answer['answer_start']
          answer_stop = answer_start + len(answer_raw)
          if passage[answer_start:answer_stop] != answer_raw:
            raise RuntimeError()
          word_ixs = get_containing_spans(inv, answer_start, answer_stop)
          answer_spans.append((word_ixs[0], word_ixs[-1]))
          answers_text.append(answer_raw)

        questions.append(SquadQuestion(
          question["id"], [x["word"] for x in q_tokens],
          answers_text, np.array(answer_spans, dtype=np.int32),
        ))

      out.append(AnnotatedSquadParagraph(
        passage, words, inv, pos, ner, sentence_lens, questions))
    return out


def cache_docs(source_file, output_file, port, n_processes):
  annotator = SquadAnnotator(port, True)
  with open(source_file, "r") as f:
    docs = json.load(f)["data"]

  paragraphs = py_utils.flatten_list(x["paragraphs"] for x in docs)

  annotated = process_par(paragraphs, annotator, n_processes, 30, "annotate")

  with open(output_file, "wb") as f:
    pickle.dump(annotated, f)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("source_file", help="SQuAD source file")
  parser.add_argument("output_file", help="Output pickle file to dump the annotated paragraphs")
  parser.add_argument("--port", type=int, default=9000, help="CoreNLP port")
  parser.add_argument("--n_processes", "-n", type=int, default=1)
  args = parser.parse_args()
  cache_docs(args.source_file, args.output_file, args.port, args.n_processes)


if __name__ == "__main__":
  main()
