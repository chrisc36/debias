import numpy as np
import tensorflow as tf
from tqdm import tqdm

from debias.utils import ops
from triviaqa_cp.triviaqa_cp_evaluation import normalize_answer, f1_score


def eval_triviaqa(predicted_ans, actual_answers, use_tqdm=False):
  """
  :param predicted_ans: List of strings,
  :param actual_answers: List of list of strings
  :param use_tqdm: Show progress with tqdm
  :return: ndarray of size [n_answers, 2] with the em/f1 scores
  """
  scores = np.zeros((len(predicted_ans), 2), dtype=np.float32)
  if use_tqdm:
    it = tqdm(list(zip(predicted_ans, actual_answers)), ncols=100, desc="eval")
  else:
    it = zip(predicted_ans, actual_answers)
  for i, (predicted_ans, actual) in enumerate(it):
    predicted = predicted_ans
    em = 0
    f1 = 0
    if len(actual) > 0:
      predicted = normalize_answer(predicted)
      for ans in actual:
        if len(ans) == 0:
          continue
        em = em or (ans == predicted)
        f1 = max(f1, f1_score(predicted, ans))
    else:
      em = len(predicted) == 0
      f1 = len(predicted) == 0
    scores[i] = (em, f1)

  return scores


def _eval_triviaqa_decode(predicted_ans, actual_answers):
  return eval_triviaqa(
    [x.decode("utf-8") for x in predicted_ans],
    [[x.decode("utf-8") for x in ans if len(x) > 0] for ans in actual_answers]
  )


def eval_triviaqa_op(logits, tokens, actual_answers, bound):
  """Tensorflow op to compute em/f1 scores using TriviaQA metrics"""
  answer_spans = ops.get_best_span(logits, bound)

  # Unlike SQuAD, for TriviaQA we don't bother properly untokenizing the
  # span, and just return the tokens with space seperators, since
  # that is almost always good enough for TriviaQA
  answer_text = tf.map_fn(
    lambda i: tf.reduce_join(tokens[i, answer_spans[i][0]:answer_spans[i][1] + 1], 0, separator=" "),
    tf.range(ops.get_shape_tuple(logits, 0)),
    dtype=tf.string, back_prop=False
  )
  scores = tf.py_func(_eval_triviaqa_decode, [answer_text, actual_answers], tf.float32, False)
  scores.set_shape([logits.shape.as_list()[0], 2])
  return scores

