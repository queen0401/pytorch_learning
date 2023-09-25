import torch
from transformers import BertModel, BertTokenizer, BertConfig

tokenizer =BertTokenizer.from_pretrained('bert-base-chinese')
# config = BertConfig.from_pretrained('bert-base-chinese')
# config.update({'output_hidden_states':True})
# model = BertModel.from_pretrained('bert-base-chinese',config = config)

from transformers import AutoModel
checkpoint = 'bert-base-chinese'
model = AutoModel.from_pretrained(checkpoint)

