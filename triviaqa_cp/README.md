# TriviaQA-CP
This directory contains information and supporting links about two datasets: 
TriviaQA-CP Location and TriviaQA-CP Person. These datasets are intended to test extractive QA models
on data that undergoes a significant train/test domain-shift.

These datasets were built from [TriviaQA](https://nlp.cs.washington.edu/triviaqa/). 
In particular, TriviaQA questions were classified as being about 'person', 'location', or 'other' entities, and
a context paragraph was pre-selected for each question.
Then location dataset was constructed by using location questions in the TriviaQA dev set as a test set,
and person/other questions in the TriviaQA train set as the train set.
The person dataset is built in the same way for person questions. 
See our the appendix of our [paper](https://arxiv.org/abs/1909.03683) for details.

## Data
First, download the [train](https://drive.google.com/open?id=1Qjfpyb-Y2cvwmiT7tsBQF_LqC-rWVskn) and [dev](https://drive.google.com/open?id=1mNt2GvXrra5EKmfHkQBMwpWuZAtiQ0am) TriviaQA questions 
that we have matched with question types and context passages.

The data consists of a modified version of the TriviaQA Web data, in particular each
json file contains a list of questions (in the "Data" field), with the following format:

```
{
    "QuestionId": str, The QuestionId of the TriviaQA question
    "Answer": dict, Answer from the TriviaQA question
    "Question": str, Question text from the TriviaQA question 
    "Passage": str, Selected passage for use as context
    "QuestionType": str, The question type, either "person", "location" or "other"
    "QuestionTypeProbs": list, Question type probabilities, list containing the 
                         other probabilities is other, person, location as judged by
                         our classifier
    "Document": dict, Meta-data about the document the passage is from, this is either 
                an 'EntityPage' or 'SearchResult' object from the TriviaQA question
    "CharStart": int, Character start offset of the passage in the document 
    "CharEnd": int, Character end offset of the passage in the document
}

```

To construct the actual datasets, these questions need to be selectively filtered: 

- TriviaQA Location Train: load the train data and select other/person questions
- TriviaQA Location Dev: load the dev data and select other/person questions
- TriviaQA Location Test: load the dev data and select only location questions
- TriviaQA Person Train: load the train data and select other/location questions
- TriviaQA Person Dev: load the dev data and select other/location questions
- TriviaQA Person Test: load the dev data and select only person questions

 
triviaqa_cp_loader.py contains code to do the filtering.

Note that a very small number of questions (about 1.5%) of questions were filtered from TriviaQA
because we could not find any context passage that contains an answer (we used a slightly more 
stringent answer-detection algorithm then the original dataset). The downloaded
dataset are thus subsets of the "web-train" and "web-dev" questions from the TriviaQA. 
We don't use the verified questions since, after filtering, there would be too few to examples
to do a statistically significant evaluation.

## Evaluation
We use the same evaluation protocol as TriviaQA, triviaqa_cp_evaluation.py 
contains a slightly modified script that can be used to evaluate
a dictionary of predictions.

## Out-of-Domain Dev Sets
It is important to note that, unfortunately, these datasets do not have out-of-domain dev sets. 
It might be tempting to use the filtered questions from the train set as a dev set, but since our goal is to simulate a 
domain-shift for which we have no examples for, that would still be a form of cheating.

In hindsight, the right approach would be to develop on the person split and then test using the location split. 
I encourage users to do this, but I can't say we did this in our experiments.

In general please be cognizant on the risk over-fitting and use the test sets with caution.