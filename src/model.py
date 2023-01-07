import transformers
import torch.nn as nn
import config

class BERTBaseUncased(nn.Module):
    def _init_(self):
        super(BERTBaseUncased.self)._init_()
        self.bert =transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768,1)
        
    def forward(self,ids,mask,token_type_ids):
        _,o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        bo=self.bert_drop(o2)
        output=self.out(bo)
        return output
        
            