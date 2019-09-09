from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn

from debias.bert.clf_debias_loss_functions import ClfDebiasLossFunction


class BertWithDebiasLoss(BertPreTrainedModel):
  """Pre-trained BERT model that uses our loss functions"""

  def __init__(self, config, num_labels, loss_fn: ClfDebiasLossFunction):
    super(BertWithDebiasLoss, self).__init__(config)
    self.num_labels = num_labels
    self.loss_fn = loss_fn
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, num_labels)
    self.apply(self.init_bert_weights)

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, bias=None):
    _, pooled_output = self.bert(
      input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    logits = self.classifier(self.dropout(pooled_output))
    if labels is None:
      return logits
    loss = self.loss_fn.forward(pooled_output, logits, bias, labels)
    return logits, loss
