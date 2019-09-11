import json


def get_qtypes(dataset_name, part):
  """Return list of question-types for a particular TriviaQA-CP dataset"""
  if dataset_name not in {"location", "person"}:
    raise ValueError("Unknown dataset %s" % dataset_name)

  if part not in {"train", "dev", "test"}:
    raise ValueError("Unknown part %s" % part)

  is_biased = part in {"train", "dev"}
  is_location = dataset_name == "location"

  if is_biased and is_location:
    return ["person", "other"]
  elif not is_biased and is_location:
    return ["location"]
  elif is_biased and not is_location:
    return ["location", "other"]
  elif not is_biased and not is_location:
    return ["person"]
  else:
    raise RuntimeError()


def load_triviaqa_cp(filename, dataset_name, part, expected_version=None):
  """Load a TriviaQA-CP dataset

  :param filename: The TriviaQA-CP train or dev json file, must be the train file if
                   if `part`=="train" and the dev file otherwise
  :param dataset_name: dataset to load, must be in ["person", "location"]
  :param part: which part, must be in ["test", "dev", "train"[
  :param expected_version: Optional version to require the data to match
  :return: List of question in dictionary form
  """
  target_qtypes = get_qtypes(dataset_name, part)

  with open(filename, "r") as f:
    data = json.load(f)

  if expected_version is not None:
    if expected_version != data["Version"]:
      raise ValueError("Expected version %s, but data was version %s" % (
        expected_version, data["Version"]))

  if part == "train":
    if data["Split"] != "Train":
      raise ValueError("Expected train file, but split is %s" % data["Split"])
  else:
    if data["Split"] != "Dev":
      raise ValueError("Expected dev file, but split is %s" % data["Split"])

  out = []
  for question in data["Data"]:
    if question["QuestionType"] in target_qtypes:
      out.append(question)
  return out
