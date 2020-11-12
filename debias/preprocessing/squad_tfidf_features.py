import pickle
from os import mkdir
from os.path import join, exists

from tqdm import tqdm

from debias.config import SQUAD_TFIDF_FEATURES, SQUAD_FILTERED_TFIDF_FEATURES
from debias.datasets.squad import load_annotated_squad, AnnotatedSquadParagraph
from debias.utils import py_utils
import numpy as np


def get_squad_tfidf_features(dataset_name, pos_filter):
  """Gets the tfidf features we used to train the bias-only model

  :param dataset_name: Name of SQuAD dataset to get features for
  :param pos_filter: POS filtered TF-IDF scores or not
  :return: Dictionary of question_id -> per-word array of TT-IDF scores
  """
  if pos_filter:
    root = SQUAD_FILTERED_TFIDF_FEATURES
  else:
    root = SQUAD_TFIDF_FEATURES
  src = join(root, dataset_name + ".pkl")
  if not exists(src):
    build_squad_tfidf_features(pos_filter)
  return py_utils.load_pickle(src)


def get_pos_filtered_sentences(ex: AnnotatedSquadParagraph):
  on_ix = 0
  pruned = []
  for l in ex.sentence_lens:
    tok = ex.tokens[on_ix:on_ix + l]
    pos = ex.pos_tags[on_ix:on_ix + l]
    on_ix += l
    pruned.append([w for w, p in zip(tok, pos) if not (p == "CD" or p.startswith("NNP"))])
  return pruned


def build_squad_tfidf_features(pos_filter=False):
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import cosine_similarity

  if pos_filter:
    print("Computing SQuAD pos-filtered TF-IDF Features")
  else:
    print("Computing SQuAD TF-IDF Features")
  print("Loading train...")
  train = load_annotated_squad("train")

  print("Fitting training...")
  tfidf = TfidfVectorizer(
    strip_accents="unicode",
    ngram_range=(1, 3), max_df=0.3, min_df=50,
    tokenizer=lambda x: x,
    token_pattern=None,
    preprocessor=lambda x: [w.lower() for w in x]
  )
  if pos_filter:
    text = py_utils.flatten_list(get_pos_filtered_sentences(ex) for ex in train)
  else:
    text = py_utils.flatten_list(ex.sentences() for ex in train)
  tfidf.fit(text)

  to_eval = [
    ("train", train),
    ("dev", None),
    ("add_sent", None),
    ("add_one_sent", None)
  ]

  if pos_filter:
    root = SQUAD_FILTERED_TFIDF_FEATURES
  else:
    root = SQUAD_TFIDF_FEATURES
  if not exists(root):
    mkdir(root)

  for ds_name, data in to_eval:
    print("Building features for " + ds_name)
    if data is None:
      data = load_annotated_squad(ds_name)
    features = {}
    for ex in tqdm(data, ncols=100, desc=ds_name):
      if pos_filter:
        sentences = get_pos_filtered_sentences(ex)
      else:
        sentences = ex.sentences()
      token_to_sent_id = np.zeros(len(ex.tokens), np.int32)
      on = 0
      for i, l in enumerate(ex.sentence_lens):
        token_to_sent_id[on:on+l] = i
        on += l

      question_fe = tfidf.transform([x.words for x in ex.questions], copy=None)
      sentence_fe = tfidf.transform(sentences, copy=None)
      question_to_sentence_dist = cosine_similarity(question_fe, sentence_fe)

      for q_ix, q in enumerate(ex.questions):
        features[q.question_id] = question_to_sentence_dist[q_ix][token_to_sent_id]

    src = join(root, ds_name + ".pkl")
    with open(src, "wb") as f:
      pickle.dump(features, f)
  print('Done!')





