# Ensemble Based Debiasing
This repo contains the code for our paper ["Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases"](TDB).
In particular, it contains code to train various models that are debiased, meaning they are trained to 
avoid using particular strategies that are known to work well on the training data, but do not generalize to
out-of-domain of adversarial settings. 

Code for our VQA experiments is in a separate [repo](https://github.com/chrisc36/bottom-up-attention-vqa).

Details and links to the TriviaQA-CP dataset we constructed are in the triviaqa_cp folder.
## Overview
### Tasks
This repo contains code to run our debiasing-methods on four test cases:

1. [MNLI](https://www.nyu.edu/projects/bowman/multinli/paper.pdf) modified to contain a synthetic bias 
2. [MNLI](https://www.nyu.edu/projects/bowman/multinli/paper.pdf) with [HANS](https://arxiv.org/abs/1902.01007) as the test set
3. [SQuAD](https://arxiv.org/abs/1606.05250) with [Adversarial SQuAD]() as the test set
4. The TriviaQA-CP datasets, which we construct from [TriviaQA](https://arxiv.org/abs/1705.03551)

Our VQA experiments are in separate [repo](https://github.com/chrisc36/bottom-up-attention-vqa).

### Code
Our implementation exists in the debias folder, and uses tensorflow 1.13.1. 

The MNLI task has an alternative, BERT implementation using pytorch in debias/bert.

Details and download links for the dataset we constructed, TriviaQA-CP, can be found in the triviaqa_cp folder. 

The actual implementation of the ensemble loss functions can be found in three places:

1. `debias/modules/clf_debias_loss_functions.py` for tensorflow classification models
2. `debias/modules/qa_debias_loss_functions.py` for tensorflow QA models
3. `debias/bert/clf_debias_loss_functions.py` for pytorch classification models


## Setup
### Dependencies
We require python>=3.6 and tensorflow 1.13.1. Additional requirements are are in

`requirements.txt`

To install, make sure tensorflow 1.13.1 is installed, then run:

`pip3 install -r requirements.txt`

The bert implementation additionally requires pytorch 1.1.0, and the 
[hugging-face pre-trained transformer module](https://github.com/huggingface/pytorch-transformers).

### Data
Scripts will automatically 
download any data they need. See config.py for the download locations, by default
everything will be downloaded to ./data.
The first time models are run be patient, some of the downloads can take a while.

To download everything beforehand, run `python debias/download_all.py`

All the data takes about 2.1G.

### Setup Example
On a fresh machine with Ubuntu 18.04, I got the tensorflow code running by installing [Cuda 10.0](https://developer.nvidia.com/cuda-10.0-download-archive?), 
[Cudnn 7.6.2](https://developer.nvidia.com/rdp/cudnn-archive), 
and running:

```
sudo apt install pip3
pip3 install tensorflow-gpu==1.13.1
pip3 install -r requirements.txt
```

## Running Experiments
Each task has a corresponding script in `debias/experiments/train_debiased_*`. 
Scripts take command line options
to specify which debiasing method to use, and sometimes additional options to specify
different variations of the task. 

For example, to train a model SQuAD with the TFIDF Filtered bias:

`python debias/experiments/train_debiased_squad.py --bias tfidf_filtered --output_file /path/to/output`

Or to train a model on the TriviaQA-CP Location dataset with the Reweight method:

`python debias/experiments/train_debiased_triviaqa_cp.py --dataset location 
--output_dir /path/to/output --mode reweight`

See the command line options (e.i., `python debias/experiments/train_debiased_squad.py --help`)
for additional options. 

Model are automatically evaluated after training, but can be re-evaluated using the evaluation scripts `debias/experiments/eval_*`

The BERT for HANS has its own script, its does not require tensorflow to be installed,
but does require pytorch 1.1 and [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-transformers)
It can be run by:

`python debias/bert/train_bert.py --do_train --do_eval --output_dir /path/to/output`

I highly recommend using the `--fp16` flag as well if you have [apex](https://github.com/NVIDIA/apex) installed, its about 2x faster.


Results should match the numbers in our paper, although please note that individual runs have 
a moderately high variance. 

## Preprocessing
Our SQuAD and TriviaQA models require data to be pre-processed using CoreNLP, and 
most experiments require pre-training a bias-only model.
The experiment scripts will download cached pre-processed data instead
of re-building them to help ensure our main experiments are easy to reproduce,
so they do NOT require completing these steps yourself.

We will upload code to do this pre-processing when ready.

## Cite
If you use this work, please cite:

"Don’t Take the Easy Way Out: Ensemble Based Methods for Avoiding Known Dataset Biases". 
Christopher Clark, Mark Yatskar, Luke Zettlemoyer. In EMNLP 2019.