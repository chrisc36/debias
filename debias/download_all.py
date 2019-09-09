from debias.datasets import squad, mnli, triviaqa_cp
from debias.utils import py_utils, load_word_vectors


def download_squad():
  squad.load_bias("train")

  for dataset in ["train", "dev", "add_sent", "add_sent_one"]:
    squad.load_annotated_squad(dataset)

  for dataset in ["dev", "add_sent", "add_sent_one"]:
    squad.load_squad_documents(dataset)


def download_triviaqa_cp():
  for dataset in ["location", "person"]:
    triviaqa_cp.load_bias(dataset)
    triviaqa_cp.load_triviaqa_cp(dataset, "test")

  for is_train in [True, False]:
    triviaqa_cp.load_annotated_triviaqa(is_train)


def download_mnli():
  mnli.ensure_mnli_is_downloaded()
  mnli.load_hans()
  mnli.load_bias("train")


def download_wordvecs():
  load_word_vectors.download_fasttext()
  load_word_vectors.download_glove_6b()


def main():
  py_utils.add_stdout_logger()
  download_wordvecs()
  download_mnli()
  download_squad()


if __name__ == '__main__':
  main()
