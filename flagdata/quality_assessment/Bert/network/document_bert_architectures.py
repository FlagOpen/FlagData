import torch
import time
from torch import nn
from torch.nn import LSTM
from transformers import BertPreTrainedModel, BertConfig, BertModel
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(7)


class DocumentBertSentenceChunkAttentionLSTM(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig, pretrained_bert=None):
        super(DocumentBertSentenceChunkAttentionLSTM, self).__init__(bert_model_config)
        if pretrained_bert is None:
            self.bert = BertModel(bert_model_config)
        else:
            self.bert = pretrained_bert
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.lstm = LSTM(bert_model_config.hidden_size, bert_model_config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size, 1)
        )
        self.w_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, bert_model_config.hidden_size))
        self.b_omega = nn.Parameter(torch.Tensor(1, bert_model_config.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(bert_model_config.hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.b_omega, -0.1, 0.1)
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor, bert_batch_size=0):
        # if self.training:
        #     # 不同doc串行，为了更大的batch_size
        #     bert_output = torch.zeros(
        #         size=(document_batch.shape[0], min(document_batch.shape[1],bert_batch_size),
        #               self.bert.config.hidden_size), dtype=torch.float, device=self.device)

        #     for doc_id in range(document_batch.shape[0]):
        #         bert_output[doc_id][:bert_batch_size] = self.dropout(
        #             self.bert(document_batch[doc_id][:bert_batch_size, 0],
        #                       token_type_ids=document_batch[doc_id][:bert_batch_size, 1],
        #                       attention_mask=document_batch[doc_id][:bert_batch_size, 2])[1])
        # else:
        # 并行化推理
        N = document_batch.shape[0]
        bert_batch_size = min(bert_batch_size, document_batch.shape[1])
        bert_output = self.dropout(
            self.bert(
                document_batch[:, :bert_batch_size, 0, :].reshape((N * bert_batch_size, -1)),
                token_type_ids=document_batch[:, :bert_batch_size, 1, :].reshape((N * bert_batch_size, -1)),
                attention_mask=document_batch[:, :bert_batch_size, 2, :].reshape((N * bert_batch_size, -1))
            )[1])
        bert_output = bert_output.reshape((N, bert_batch_size, -1))

        output, (_, _) = self.lstm(bert_output.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        # (batch_size, seq_len, num_hiddens)
        attention_w = torch.tanh(torch.matmul(output, self.w_omega) + self.b_omega)
        attention_u = torch.matmul(attention_w, self.u_omega)  # (batch_size, seq_len, 1)
        attention_score = F.softmax(attention_u, dim=1)  # (batch_size, seq_len, 1)
        attention_hidden = output * attention_score  # (batch_size, seq_len, num_hiddens)
        attention_hidden = torch.sum(attention_hidden, dim=1)  # 加权求和 (batch_size, num_hiddens)
        prediction = self.mlp(attention_hidden)
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction


class DocumentBertCombineWordDocumentLinear(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig, pretrained_bert=None):
        super(DocumentBertCombineWordDocumentLinear, self).__init__(bert_model_config)
        if pretrained_bert is None:
            self.bert = BertModel(bert_model_config)
        else:
            self.bert = pretrained_bert
        self.bert_batch_size = 1  # 为了运算速度，对于不同的文章仅取第一个分段
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)

        self.mlp = nn.Sequential(
            nn.Dropout(p=bert_model_config.hidden_dropout_prob),
            nn.Linear(bert_model_config.hidden_size * 2, 1)
        )
        self.mlp.apply(init_weights)

    def forward(self, document_batch: torch.Tensor):
        # if self.training:
        #     # 不同doc串行，为了更大的batch_size
        #     bert_output = torch.zeros(size=(document_batch.shape[0],
        #                                     min(document_batch.shape[1], self.bert_batch_size),
        #                                     self.bert.config.hidden_size * 2),
        #                             dtype=torch.float, device=self.device)

        #     for doc_id in range(document_batch.shape[0]):
        #         all_bert_output_info = self.bert(document_batch[doc_id][:self.bert_batch_size, 0].squeeze(dim=1),
        #                                         token_type_ids=document_batch[doc_id][:self.bert_batch_size, 1],
        #                                         attention_mask=document_batch[doc_id][:self.bert_batch_size, 2])
        #         bert_token_max = torch.max(all_bert_output_info[0], 1)
        #         bert_output[doc_id][:self.bert_batch_size] = torch.cat(
        #             (bert_token_max.values, all_bert_output_info[1]), 1)
        # else:
        # 并行化推理
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        all_bert_output_info = self.bert(document_batch[:, :self.bert_batch_size, 0, :].squeeze(dim=1),
                                         token_type_ids=document_batch[:, :self.bert_batch_size, 1, :].squeeze(dim=1),
                                         attention_mask=document_batch[:, :self.bert_batch_size, 2, :].squeeze(dim=1))
        # end.record()
        # torch.cuda.synchronize()
        # print("BERT-DOC-TOKEN 耗时" + str(start.elapsed_time(end)))

        bert_token_max = torch.max(all_bert_output_info[0], 1)
        bert_output = torch.cat((bert_token_max.values, all_bert_output_info[1]), 1)

        prediction = self.mlp(bert_output.view(bert_output.shape[0], -1))
        assert prediction.shape[0] == document_batch.shape[0]
        return prediction
