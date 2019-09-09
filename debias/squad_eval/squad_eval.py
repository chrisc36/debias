import numpy as np
import tensorflow as tf
from tqdm import tqdm

from debias.squad_eval.squad_v1_official_evaluation import normalize_answer, f1_score
from debias.utils.ops import get_best_span


def eval_squad(predicted_ans, actual_answers, use_tqdm=False):
  """
  :param predicted_ans: List of strings,
  :param actual_answers: List of list of strings
  :param use_tqdm: Show progress with tqdm
  :return: ndarray of size [n_answers, 2] with the em/f1 scores
  """
  if use_tqdm:
    it = tqdm(list(zip(predicted_ans, actual_answers)), ncols=100, desc="eval")
  else:
    it = zip(predicted_ans, actual_answers)

  scores = np.zeros((len(predicted_ans), 2), dtype=np.float32)
  for i, (predicted_ans, actual) in enumerate(it):
    em = 0
    f1 = 0
    if len(actual) > 0:
      predicted_ans = normalize_answer(predicted_ans)
      for ans in actual:
        if len(ans) == 0:
          continue
        ans = normalize_answer(ans)
        em = em or (ans == predicted_ans)
        f1 = max(f1, f1_score(predicted_ans, ans))
    else:
      em = len(predicted_ans) == 0
      f1 = len(predicted_ans) == 0
    scores[i] = (em, f1)

  return scores


def _eval_squad_from_spans(spans, invs, texts, actual_answers):
  predicted_answers = []
  actual_answers = [[x.decode("utf-8") for x in ans if len(x) > 0] for ans in actual_answers]
  for i in range(len(spans)):
    inv = invs[i]
    text = texts[i].decode("utf-8")
    ans = text[inv[spans[i, 0], 0]:inv[spans[i, 1], 1]]
    predicted_answers.append(ans)
  return eval_squad(predicted_answers, actual_answers)


def eval_squad_op(span_logits, inv, passage_text, actual_answers, max_bound):
  """Tensorflow op to compute em/f1 scores using SQuAD metrics"""
  batch = inv.shape.as_list()[0]
  predicted_span = get_best_span(span_logits, max_bound)
  scores = tf.py_func(
    _eval_squad_from_spans,
    [predicted_span, inv, passage_text, actual_answers],
    tf.float32, False)
  scores.set_shape([batch, 2])
  return scores

