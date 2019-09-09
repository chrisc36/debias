from argparse import ArgumentParser

from debias.modules import clf_debias_loss_functions
from debias.modules import qa_debias_loss_functions


def add_general_args(parser: ArgumentParser):
  """Arguments that are common between all experiments"""
  parser.add_argument("--output_dir", "-o", default=None,
                      help="Place to store the model")
  parser.add_argument("--n_processes", "-n", type=int, default=4,
                      help="Number of processes to use when pre-processing")
  parser.add_argument("--debug", "--dbg", action="store_true",
                      help="Run on smaller model on a sample of the data")


def add_loss_args(argparser: ArgumentParser, default_penalty):
  """Arguments for selecting the loss function"""
  argparser.add_argument("--mode", choices=["bias_product", "none", "learned_mixin", "reweight"],
                         default="learned_mixin", help="Kind of debiasing method to use")
  argparser.add_argument("--penalty", type=float, default=default_penalty,
                         help="Penalty weight for the learn_mixin model")


def get_clf_loss_fn(args) -> clf_debias_loss_functions.ClfDebiasLossFunction:
  if args.mode == "none":
    fn = clf_debias_loss_functions.Plain()
  elif args.mode == "reweight":
    fn = clf_debias_loss_functions.Reweight()
  elif args.mode == "bias_product":
    fn = clf_debias_loss_functions.BiasProduct()
  elif args.mode == "learned_mixin":
    fn = clf_debias_loss_functions.LearnedMixin(args.penalty)
  else:
    raise RuntimeError()
  return fn


def get_qa_loss_fn(args) -> qa_debias_loss_functions.QaDebiasLossFunction:
  if args.mode == "none":
    fn = qa_debias_loss_functions.Plain()
  elif args.mode == "reweight":
    fn = qa_debias_loss_functions.Reweight()
  elif args.mode == "bias_product":
    fn = qa_debias_loss_functions.BiasProduct()
  elif args.mode == "learned_mixin":
    fn = qa_debias_loss_functions.LearnedMixin(args.penalty)
  else:
    raise RuntimeError()
  return fn
