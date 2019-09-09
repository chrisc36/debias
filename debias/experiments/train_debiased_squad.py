import argparse
import logging

# Need to import there for pickle
from debias.datasets.dataset_utils import QuantileBatcher
from debias.datasets.squad import AnnotatedSquadLoader
from debias.experiments.eval_debiased_squad import compute_all_scores
from debias.models.text_pair_qa_model import TextPairQaDebiasingModel
from debias.modules.attention_layers import WeightedDot, BiAttention
from debias.modules.cudnn_recurrent_dropout import CudnnLSTMRecurrentDropout
from debias.modules.layers import VariationalDropout, seq, FullyConnected, MaxPooler, Conv1d
from debias.modules.word_and_char_encoder import WordAndCharEncoder
from debias.training.evaluator import Evaluator
from debias.training.trainer import Trainer, AdamOptimizer
from debias.utils import py_utils, cli_utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--stratify", type=int, default=None)
  parser.add_argument("--bias", choices=["tfidf", "tfidf_filtered"], default="tfidf_filtered")
  cli_utils.add_general_args(parser)
  cli_utils.add_loss_args(parser, default_penalty=2.0)
  args = parser.parse_args()

  if args.stratify is None:
    if args.mode == "learned_mixin":
      # Note sure if this actually makes a difference, but I turned this on
      # for the learned_mixin case so we do here for exactness
      args.stratify = 6

  dbg = args.debug

  if dbg:
    epoch_size = 50
  else:
    epoch_size = 1341

  opt = AdamOptimizer(max_grad_norm=5.0)
  batcher = QuantileBatcher(45, 10, 300, 4, 12)
  evaluator = Evaluator("squad")

  trainer = Trainer(
    batcher, opt, evaluator,
    eval_batch_size=90,
    num_epochs=30, epoch_size=epoch_size,
    log_period=100,
    prefetch=5, loss_ema=0.999,
    n_processes=args.n_processes
  )

  filtered_bias = args.bias == "tfidf_filtered"
  if dbg:
    dataset = AnnotatedSquadLoader(
      sample_train=1000, sample_dev=500, stratify=args.stratify, filtered_bias=filtered_bias)
  else:
    dataset = AnnotatedSquadLoader(
      sample_train_eval=10000, stratify=args.stratify, filtered_bias=filtered_bias)

  dim = 100
  recurrent_layer = CudnnLSTMRecurrentDropout(dim, 0.0)
  model = TextPairQaDebiasingModel(
    None,  # Assume pre-tokenized data
    text_encoder=WordAndCharEncoder(
      "glove.6B.50d" if dbg else "crawl-300d-2M",
      first_n=None,
      char_embed_dim=24,
      character_mapper=Conv1d(100, 5, None),
      character_pooler=MaxPooler(),
      word_length=30
    ),
    map_embed=seq(
      VariationalDropout(0.2),
      recurrent_layer,
      VariationalDropout(0.2)
    ),
    fuse_layer=BiAttention(WeightedDot()),
    post_process_layer=seq(
      FullyConnected(dim * 2, activation="glu"),
      VariationalDropout(0.2),
      recurrent_layer,
      VariationalDropout(0.2),
      recurrent_layer,
      VariationalDropout(0.2),
    ),
    debias_loss_fn=cli_utils.get_qa_loss_fn(args)
  )

  with open(__file__) as f:
    notes = f.read()

  py_utils.add_stdout_logger()

  trainer.train(dataset, model, args.output_dir, notes)

  if args.output_dir:
    logging.info("Evaluating")
    compute_all_scores(args.output_dir, ["dev", "add_sent", "add_one_sent"])


if __name__ == '__main__':
  main()
