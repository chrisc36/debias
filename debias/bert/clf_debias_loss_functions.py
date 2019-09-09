import torch
from torch import nn
from torch.nn import functional as F


class ClfDebiasLossFunction(nn.Module):
  """Torch classification debiasing loss function"""

  def forward(self, hidden, logits, bias, labels):
    """
    :param hidden: [batch, n_features] hidden features from the model
    :param logits: [batch, n_classes] logit score for each class
    :param bias: [batch, n_classes] log-probabilties from the bias for each class
    :param labels: [batch] integer class labels
    :return: scalar loss
    """
    raise NotImplementedError()


class Plain(ClfDebiasLossFunction):
  def forward(self, hidden, logits, bias, labels):
    return F.cross_entropy(logits, labels)


class ReweightByInvBias(ClfDebiasLossFunction):
  def forward(self, hidden, logits, bias, labels):
    logits = logits.float() # In case we were in fp16 mode
    loss = F.cross_entropy(logits, labels, reduction='none')
    one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
    weights = 1 - (one_hot_labels * torch.exp(bias)).sum(1)
    return (weights * loss).sum() / weights.sum()


class BiasProduct(ClfDebiasLossFunction):
  def forward(self, hidden, logits, bias, labels):
    logits = logits.float()  # In case we were in fp16 mode
    logits = F.log_softmax(logits, 1)
    return F.cross_entropy(logits+bias.float(), labels)


class LearnedMixin(ClfDebiasLossFunction):

  def __init__(self, penalty):
    super().__init__()
    self.penalty = penalty
    self.bias_lin = torch.nn.Linear(768, 1)

  def forward(self, hidden, logits, bias, labels):
    logits = logits.float()  # In case we were in fp16 mode
    logits = F.log_softmax(logits, 1)

    factor = self.bias_lin.forward(hidden)
    factor = factor.float()
    factor = F.softplus(factor)

    bias = bias * factor

    bias_lp = F.log_softmax(bias, 1)
    entropy = -(torch.exp(bias_lp) * bias_lp).sum(1).mean(0)

    loss = F.cross_entropy(logits + bias, labels) + self.penalty*entropy
    return loss

