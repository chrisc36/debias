import argparse
import logging

# Need to import there for pickle
from debias.datasets.dataset_utils import QuantileBatcher
from debias.datasets.triviaqa_cp import AnnotatedTriviaQACPLoader
from debias.experiments import eval_debiased_triviaqa_cp
from debias.models.text_pair_qa_model import TextPairQaDebiasingModel
from debias.modules.attention_layers import WeightedDot, BiAttention
from debias.modules.cudnn_recurrent_dropout import CudnnLSTMRecurrentDropout
from debias.modules.layers import VariationalDropout, seq, FullyConnected, MaxPooler, Conv1d, \
  Dropout, HighwayLayer
from debias.modules.word_and_char_encoder import WordAndCharEncoder
from debias.training.evaluator import Evaluator
from debias.training.trainer import Trainer, AdamOptimizer
from debias.utils import py_utils, cli_utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--stratify", type=int, default=None)
  parser.add_argument("--dataset", choices=["location", "person"], default="location")
  cli_utils.add_general_args(parser)
  cli_utils.add_loss_args(parser, default_penalty=None)
  args = parser.parse_args()

  if args.stratify is None:
    if args.mode == "learned_mixin":
      # Note sure if this actually makes a difference, but I turned this on
      # for the learned_mixin case so we do here for exactness
      args.stratify = 6

  if args.penalty is None:
    if args.dataset == "person":
      args.penalty = 0.2
    else:
      args.penalty = 0.4

  dbg = args.debug

  if dbg:
    epoch_size = 50
  else:
    epoch_size = 1200

  opt = AdamOptimizer(decay_steps=50, max_grad_norm=3.0)
  batcher = QuantileBatcher(45, 10, 400, 4, 12)
  evaluator = Evaluator("triviaqa")

  trainer = Trainer(
    batcher, opt, evaluator,
    eval_batch_size=90,
    num_epochs=30, epoch_size=epoch_size,
    log_period=100,
    prefetch=5, loss_ema=0.999,
    n_processes=args.n_processes
  )

  if dbg:
    dataset = AnnotatedTriviaQACPLoader(
      args.dataset, sample_train=1000, stratify=args.stratify)
  else:
    dataset = AnnotatedTriviaQACPLoader(
      args.dataset, sample_train_eval=8000, stratify=args.stratify)

  dim = 128
  recurrent_layer = CudnnLSTMRecurrentDropout(dim, 0.2)
  model = TextPairQaDebiasingModel(
    None,  # Assume pre-tokenized data
    text_encoder=WordAndCharEncoder(
      "glove.6B.50d" if dbg else "crawl-300d-2M",
      first_n=500000,
      char_embed_dim=24,
      character_mapper=Conv1d(100, 5, None),
      character_pooler=MaxPooler(),
      word_length=30
    ),
    map_embed=seq(
      Dropout(0.3),
      HighwayLayer(recurrent_layer),
    ),
    fuse_layer=BiAttention(WeightedDot()),
    post_process_layer=seq(
      VariationalDropout(0.2),
      FullyConnected(dim * 2, activation="relu"),
      VariationalDropout(0.2),
      HighwayLayer(recurrent_layer),
      VariationalDropout(0.2),
      HighwayLayer(recurrent_layer),
      VariationalDropout(0.2),
    ),
    debias_loss_fn=cli_utils.get_qa_loss_fn(args)
  )

  with open(__file__) as f:
    notes = f.read()

  py_utils.add_stdout_logger()

  trainer.train(dataset, model, args.output_dir, notes)

  if args.output_dir:
    logging.info("Evaluating...")
    eval_debiased_triviaqa_cp.show_scores(args.output_dir, args.dataset, ["dev", "test"])


if __name__ == '__main__':
  main()
