"""
This script builds the NER/POS tagged data we trained on for TriviaQA. It should be
run with `port` indicating were a CoreNLP server can be found, we used
`stanford-corenlp-full-2018-10-05`. If the server has multiple threads, the `n_processes`
flag can be used to send multiple queries at a time to speed up tagging.

This script does not 100% precisely reproduce the cached data that is loaded by default in our
train/eval scripts. In particular, the POS and NER tags are (very occasionally) different.
I am not sure what the cause is, but my test show <1% of the tags differ.
"""

import json
import regex
from typing import List, Union, Optional, Dict

import numpy as np
import requests

import argparse

from debias.datasets.triviaqa_cp import AnnotatedTriviaQaExample, load_annotated_triviaqa
from debias.preprocessing.corenlp_client import CoreNLPClient
from debias.utils import process_par, py_utils
from triviaqa_cp.triviaqa_cp_evaluation import normalize_answer


def extract_normalized_answers(ans):
  """Get the normalized answers from a json TriviaQa question"""

  if ans is not None:
    answers = ans['NormalizedAliases']
    human_answers = ans.get('HumanAnswers')
    if human_answers is not None:
      # This are fair game since they are used in the eval script, but be
      # careful to normalize them as well
      answers += [normalize_answer(x) for x in human_answers]
  else:
    answers = None  # test question
  return answers


def find_answer_spans(para: List[str], tokenized_answers: List[List[str]]):
  """Find spans that the eval script would given an EM of 1 in `para`"""

  words = [normalize_answer(w) for w in para]
  occurances = []
  for answer_ix, answer in enumerate(tokenized_answers):
    word_starts = [i for i, w in enumerate(words) if answer[0] == w]
    n_tokens = len(answer)
    for start in word_starts:
      end = start + 1
      ans_token = 1
      while ans_token < n_tokens and end < len(words):
        next = words[end]
        if answer[ans_token] == next:
          ans_token += 1
          end += 1
        elif next == "":
          end += 1
        else:
          break
      if n_tokens == ans_token:
        occurances.append((start, end))
  return list(set(occurances))


resplit = r"\p{Pd}\p{Po}\p{Ps}\p{Pe}\p{S}\p{Pc}"
resplit = "([" + resplit + "]|'')"
split_regex = r"(?![\.,'])" + resplit
split_regex = regex.compile(split_regex)


def extract_tokens(annotations, tags=True):
  """Extract tokens from CoreNLP output"""

  words, pos, ner = [], [], []
  sentence_lens = []
  on_len = 0
  for sentences in annotations:
    if len(sentences["tokens"]) == 0:
      raise RuntimeError()
    for token in sentences["tokens"]:
      w = token["originalText"]

      if w == "''" or w == '``':
        split = [w]
      else:
        # We tokenize a bit more aggresively the CoreNLP so span-based models
        # can make fine-grained choices of what text to return
        split = [x for x in split_regex.split(w) if len(x) > 0]

      if len(split) == 1:
        words.append(w)
        if tags:
          p, n = token["pos"], token["ner"]
          pos.append(p)
          ner.append(n)
      else:
        words += split
        if tags:
          p, n = token["pos"], token["ner"]
          ner += [n] * len(split)
          pos += ['SEP' if split_regex.match(x) else p for x in split]

    sentence_lens.append(len(words) - on_len)
    on_len = len(words)
  if tags:
    return words, pos, ner, sentence_lens
  else:
    return words, sentence_lens


class AnnotateTriviaqaQuestions(process_par.Processor):
  """Turns JSON TriviaQA questions into `AnnotatedTriviaQaExample`"""

  def __init__(self, port, reuse_session=False, legacy_tokenization=True):
    self.port = port
    self.reuse_session = reuse_session
    self.legacy_tokenization = legacy_tokenization

  def process(self, data: List[Dict]) -> List[AnnotatedTriviaQaExample]:
    cli = CoreNLPClient(port=self.port)
    sess = None
    if self.reuse_session:
      sess = requests.Session()
      sess.trust_env = False

    out = []
    for example in data:
      q_tok = extract_tokens(cli.query_tokenize(example['Question'], sess=sess)["sentences"], False)[0]

      answers = extract_normalized_answers(example["Answer"])
      answers_tokenized = []
      for ans in answers:
        ans_tok = extract_tokens(cli.query_tokenize(ans, sess=sess)["sentences"], False)[0]
        if len(ans_tok) > 0:  # Can happen very wonky unicode answers
          answers_tokenized.append(ans_tok)

      tok = []
      pos = []
      ner = []
      for para in example['Passage'].split("\n"):
        if self.legacy_tokenization:
          # The original code did tokenization and NER separately instead of doing both
          # in one query in order to do some additional caching. Unfortunately this can slightly
          # change tagging output due to the respitting we do, so we preserve that behavior here.
          _tok = extract_tokens(cli.query_tokenize(para, sess=sess)["sentences"], False)[0]
          sentences = cli.query_ner(" ".join(_tok), sess=sess, whitespace=True)["sentences"]
        else:
          sentences = cli.query_ner(para, sess=sess)["sentences"]
        p_tok, p_pos, p_ner, _ = extract_tokens(sentences, True)
        tok += p_tok
        pos += p_pos
        ner += p_ner

      spans = np.array(find_answer_spans(tok, answers_tokenized))
      spans[:, 1] -= 1  # Switch to inclusive

      out.append(AnnotatedTriviaQaExample(
        example["QuestionId"], example["QuestionType"], np.array(example["QuestionTypeProbs"]),
        q_tok, tok, pos, ner, answers, spans))
    return out


def main():
  parser = argparse.ArgumentParser("Builds annotated TriviaQA-CP data")
  parser.add_argument("source", help="Source TriviaQa-CP file")
  parser.add_argument("output", help="Output pickle file")
  parser.add_argument("--port", default=9000, type=int)
  parser.add_argument("--n_processes", default=1, type=int)
  parser.add_argument("--no_legacy_tokenization", action="store_true",
                      help="Turn off legacy tokenization, which will more closely reproduce our"
                           " results, but might make tagging worse in rare cases")
  args = parser.parse_args()

  with open(args.source, "r") as f:
    examples = json.load(f)['Data']

  annotator = AnnotateTriviaqaQuestions(
    args.port, legacy_tokenization=not args.no_legacy_tokenization)
  output = process_par.process_par(examples, annotator, args.n_processes, 10)

  with open(args.output, "wb") as f:
    f.write(output)


if __name__ == '__main__':
  main()