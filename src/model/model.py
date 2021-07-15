import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from typing import Dict, Any


class MrcBertModel(nn.Module):
    def __init__(self, config):
        super(MrcBertModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config["bert_dir"])
        self.start_outputs = nn.Linear(config["hidden_size"], 1)
        self.end_outputs = nn.Linear(config["hidden_size"], 1)
        self.span_embedding = MultiNonLinearClassifier(config["hidden_size"] * 2, 1, config["dropout"])

    @classmethod
    def create_model(cls, filepath, device):
        model_config_path = os.path.join(filepath, "config.json")
        model_params_path = os.path.join(filepath, "model.bin")
        if os.path.exists(model_config_path) and os.path.exists(model_params_path):
            with open(model_config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
                model = cls(model_config)
                model.load_state_dict(torch.load(model_params_path, map_location=device))
                model.to(device)
                return model
        else:
            print("Not found model-related file")
            raise FileNotFoundError

    def get_model_info(self):
        return self.config

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2] 相当于在每个起始位置矩阵中的每一个起始向量后面拼一个相同的结束位置向量
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len] 进入两个线性层的输出
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits


class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
