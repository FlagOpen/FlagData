import os
import time
import torch
from transformers import BertModel
from utils.encode import encode_documents
from utils.data import asap_essay_lengths, fix_score
from transformers import BertConfig, CONFIG_NAME, BertTokenizer
from network.document_bert_architectures import DocumentBertCombineWordDocumentLinear, DocumentBertSentenceChunkAttentionLSTM



class DocumentBertScoringModel(torch.nn.Module):
    def __init__(self, args):
        super(DocumentBertScoringModel, self).__init__()
        self.args = vars(args)

        # bert config
        if os.path.exists(self.args['bert_model_path']):
            cfg_path = os.path.join(self.args['bert_model_path'], CONFIG_NAME)
            if os.path.exists(cfg_path):
                config = BertConfig.from_json_file(cfg_path)

        self.config = config
        self.prompt = int(args.prompt[1])

        # 得分范围
        self.min_score, self.max_score = config.score_range[0], config.score_range[1]
        self.sigmoid = torch.nn.Sigmoid()

        # Multi-Scale-BERT-AES 的每个Segment尺度，'90_30_130_10'
        self.chunk_sizes = self.args['chunk_sizes']
        self.use_segment_scale = config.segment_cfg["use_segment_scale"]

        # bert_batch_size 每个 Segment 尺度对应的文章切分数, 处于计算速度与统计结果的平衡，对于每第 i 个segment尺度的切分段
        # 如果超过self.bert_batch_sizes[i], 取前self.bert_batch_sizes[i]个Segment
        self.bert_batch_sizes = []
        if 0 not in self.chunk_sizes:
            for chunk_size in self.chunk_sizes:
                bert_batch_size = int(asap_essay_lengths[self.prompt] / chunk_size) + 1
                self.bert_batch_sizes.append(bert_batch_size)

        # bert tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['tokenizer_path'])
            
        # BERT-TOKEN-DOC 模型，计算文档&Token 尺度得分
        if 'pretrained_model_path' in list(self.args.keys()) and \
                os.path.exists(self.args['pretrained_model_path']):
            bert_token_doc = BertModel.from_pretrained(self.args['pretrained_model_path'], 
                                                       config=config)
        else:
            bert_token_doc = BertModel.from_pretrained("bert-base-chinese")

        self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear(
            config, bert_token_doc)
      
        if self.use_segment_scale:
            # BERT-Segment 模型， 计算每个Segment尺度的得分 
            if 'pretrained_model_path' in list(self.args.keys()) and \
                    os.path.exists(self.args['pretrained_model_path']):
                print(self.args['pretrained_model_path'])
                bert_seg = BertModel.from_pretrained(self.args['pretrained_model_path'], 
                                                    config=config)
            else:
                bert_seg = BertModel.from_pretrained("bert-base-chinese")
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM(
                config, bert_seg)
        

    
    def forward(self, document_representations_word_document, 
                document_representations_chunk_list=None):
        """
        :param document_representations_word_document: 文档&Token 尺度的表示 
            Tensor
            size: (batch_size, k, 3, L)

        :param document_representations_chunk_list: 每个Segment尺度的表示
            list of Tensor 
            size: [(batch_size, k_scale, 3, L_scale)] * scale_num

        :return: 文档&Token 尺度的得分 + 每个Segment尺度的得分
        """
        # 文档&Token 尺度的得分
        batch_predictions_word_document = self.bert_regression_by_word_document(
            document_representations_word_document)
        batch_predictions_word_document = torch.squeeze(batch_predictions_word_document)
        batch_predictions_word_chunk_sentence_doc = batch_predictions_word_document

        if self.use_segment_scale:
            # 每个Segment尺度的得分
            
            for chunk_index in range(len(self.chunk_sizes)):
                batch_document_tensors_chunk = document_representations_chunk_list[chunk_index]
                
                batch_predictions_chunk = self.bert_regression_by_chunk(
                    batch_document_tensors_chunk, bert_batch_size=self.bert_batch_sizes[chunk_index])
                
                batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
                batch_predictions_word_chunk_sentence_doc = torch.add(batch_predictions_word_chunk_sentence_doc,
                                                                    batch_predictions_chunk)

        # 最终得分设定在[a, b]之间
        score = self.min_score + (self.max_score - self.min_score) * self.sigmoid(
            batch_predictions_word_chunk_sentence_doc)

        return score




    


