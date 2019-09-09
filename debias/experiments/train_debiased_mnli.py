import argparse
import logging

from debias.datasets.dataset_utils import QuantileBatcher
from debias.datasets.mnli import MnliTrainingDataLoader
from debias.experiments import eval_debiased_mnli
from debias.models.text_pair_clf_model import TextPairClfDebiasingModel
from debias.modules.attention_layers import AttentionBiFuse, WeightedDot
from debias.modules.cudnn_recurrent_dropout import CudnnLSTMRecurrentDropout
from debias.modules.layers import VariationalDropout, seq, FullyConnected, Dropout, mseq, MaxPooler, \
  Conv1d
from debias.modules.word_and_char_encoder import WordAndCharEncoder
from debias.training.evaluator import Evaluator
from debias.training.trainer import Trainer, AdamOptimizer
from debias.utils import py_utils, cli_utils
from debias.utils.tokenizer import NltkAndPunctTokenizer


def main():
  parser = argparse.ArgumentParser()
  cli_utils.add_general_args(parser)
  cli_utils.add_loss_args(parser, default_penalty=0.03)
  args = parser.parse_args()

  dbg = args.debug

  if dbg:
    epoch_size = 200
  else:
    epoch_size = 6000

  opt = AdamOptimizer(max_grad_norm=5.0)
  batcher = QuantileBatcher(32, 10, 160, 4, 12)
  evaluator = Evaluator(mode="clf")

  trainer = Trainer(
    batcher, opt, evaluator,
    eval_batch_size=64,
    num_epochs=30, epoch_size=epoch_size,
    log_period=100,
    prefetch=5, loss_ema=0.999,
    n_processes=args.n_processes,
  )

  if dbg:
    dataset = MnliTrainingDataLoader(1000, 3000, 1000)
  else:
    dataset = MnliTrainingDataLoader(n_train_eval_sample=10000)

  dim = 50 if dbg else 200
  recurrent_layer = CudnnLSTMRecurrentDropout(dim, 0.2)
  model = TextPairClfDebiasingModel(
    NltkAndPunctTokenizer(),
    WordAndCharEncoder(
      "glove.6B.50d" if dbg else "crawl-300d-2M",
      first_n=None,
      char_embed_dim=24,
      character_mapper=mseq(Dropout(0.1), Conv1d(100, 5, None)),
      character_pooler=MaxPooler(),
      word_length=30,
    ),
    map_embed=seq(
      VariationalDropout(0.2),
      recurrent_layer
    ),
    bifuse_layer=AttentionBiFuse(WeightedDot()),
    post_process_layer=seq(
      recurrent_layer,
      VariationalDropout(0.2),
    ),
    pool_layer=MaxPooler(),
    processs_joint=mseq(
      FullyConnected(100),
      Dropout(0.2)
    ),
    n_classes=3,
    debias_loss_fn=cli_utils.get_clf_loss_fn(args)
  )

  with open(__file__) as f:
    notes = f.read()

  py_utils.add_stdout_logger()

  trainer.train(dataset, model, args.output_dir, notes)

  if args.output_dir:
    logging.info("Evaluating...")
    eval_debiased_mnli.show_scores(args.output_dir, True, True, n_processes=args.n_processes)


if __name__ == '__main__':
  main()
