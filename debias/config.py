from os.path import join, dirname

SOURCE_DIR = join(dirname(dirname(__file__)), "data")

# Directory to store HANS data
HANS_SOURCE = join(SOURCE_DIR, "hans")

# Directory for GLUE, we only store MNLI data in it
GLUE_SOURCE = join(SOURCE_DIR, "glue_data")

# Directory to store SQUAD v1.1 datasets
SQUAD_SOURCE = join(SOURCE_DIR, "squad_1.1")

# Directory to store TriviaQA-CP datasets
TRIVIAQA_CP_SOURCE = join(SOURCE_DIR, "triviaqa-cp")

# Directory to store word-vectors
WORD_VEC_SOURCE = join(SOURCE_DIR, "word-vectors")

# Directory to store SQuAD with corenlp annotations
SQUAD_CORENLP = join(SOURCE_DIR, "squad-corenlp")

# Directory to store TriviaQA-CP with corenlp annotations
TRIVIAQA_CP_CORENLP = join(SOURCE_DIR, "triviaqa-corenlp")

# Directories to store the output of the bias-only models
MNLI_WORD_OVERLAP_BIAS = join(SOURCE_DIR, "mnli-word-overlap-bias")
SQUAD_TFIDF_FILTERED_BIAS = join(SOURCE_DIR, "squad-tfidf-filtered-bias")
SQUAD_TFIDF_BIAS = join(SOURCE_DIR, "squad-tfidf-bias")
TRIVIAQA_CP_PERSON_FILTERED_BIAS = join(SOURCE_DIR, "triviaqa-cp-person-bias")
TRIVIAQA_CP_LOCATION_FILTERED_BIAS = join(SOURCE_DIR, "triviaqa-cp-location-bias")