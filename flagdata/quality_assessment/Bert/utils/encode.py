import torch, math, re, logging, json
from transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, LongformerTokenizer

log = logging.getLogger()


def encode_documents(documents: list, tokenizer: BertTokenizer, max_input_length):
    # 对英文文本进行分词
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    # BERT接受的最大文本长度为512，因此需要对文本进行截断，除了[CLS]和[SEP ]最大为512
    max_sequences_per_document = math.ceil(max(len(x) / (max_input_length - 2) for x in tokenized_documents))
    # shape = (N, S, K, L) N:文档数量；S:由于BERT参数数量限制，需要将输入文本分为 S 段，S=max_sequences_per_document;
    # K: 代表token的index， token类型， attention_mask (对于padding token 不需要 attention)，L:每个序列的最大token数
    output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, max_input_length), dtype=torch.long)
    document_seq_lengths = []
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length - 2))):
            raw_tokens = tokenized_document[i:i + (max_input_length - 2)]
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 用词汇表中的index表示每个token
            attention_masks = [1] * len(input_ids)  # attention_mask: 1表示真实token，0表示padding token

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)
            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                      torch.LongTensor(input_type_ids).unsqueeze(0),
                                                      torch.LongTensor(attention_masks).unsqueeze(0)),
                                                     dim=0)
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index + 1)
    return output, torch.LongTensor(document_seq_lengths)


def encode_document(document, tokenizer, max_len, sequences):
    output = torch.zeros(size=(sequences, 3, max_len), dtype=torch.long)
    tokenized_document = tokenizer.tokenize(document)
    for seq_index, i in enumerate(range(0, len(tokenized_document), (max_len - 2))):
        raw_tokens = tokenized_document[i:i + (max_len - 2)]
        tokens = []
        input_type_ids = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in raw_tokens:
            tokens.append(token)
            input_type_ids.append(0)

        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_masks = [1] * len(input_ids)

        while len(input_ids) < max_len:
            input_ids.append(0)
            input_type_ids.append(0)
            attention_masks.append(0)

        output[seq_index, :, :] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                             torch.LongTensor(input_type_ids).unsqueeze(0),
                                             torch.LongTensor(attention_masks).unsqueeze(0)),
                                            dim=0)
        if seq_index == sequences - 1:
            break

    return output


# 将文本切分为多个片段，每个片段以的末尾都是完整的段落/句子。
def encode_document_with_whole_segment(document, tokenizer, max_len,
                                       drop_len_threshold=200):
    represent_segs = []
    text_segs = []

    start_idx = 0
    while start_idx < len(document):
        end_idx = find_end_token(document, start_idx, max_len - 2)

        if end_idx == -1:
            end_idx = start_idx + max_len - 2

        text_segment = document[start_idx:end_idx]
        if len(text_segment) < drop_len_threshold:
            start_idx = end_idx
            continue

        text_segs.append(''.join(text_segment))

        # 将segment处理为模型的输入
        segment_tokens = tokenizer.tokenize(text_segment)
        segment_tokens = ["[CLS]"] + segment_tokens + ["[SEP]"]
        segment_token_ids = tokenizer.convert_tokens_to_ids(segment_tokens)
        segmen_type_ids = [0] * len(segment_token_ids)
        segment_attention_masks = [1] * len(segment_token_ids)

        while len(segment_token_ids) < max_len:
            segment_token_ids.append(0)
            segment_attention_masks.append(0)
            segmen_type_ids.append(0)

        segment = torch.cat((torch.LongTensor(segment_token_ids).unsqueeze(0),
                             torch.LongTensor(segmen_type_ids).unsqueeze(0),
                             torch.LongTensor(segment_attention_masks).unsqueeze(0)),
                            dim=0)

        represent_segs.append(segment.unsqueeze(0))

        start_idx = end_idx
    try:
        # 使用torch拼接segments
        represent_segs = torch.cat(represent_segs, dim=0)
    except Exception as e:
        logging.warning(f"represent_segs ERROR :  {represent_segs}")
        logging.warning(f"INFO: {e}")
    return text_segs, represent_segs


def find_end_token(tokens, start_idx, max_len):
    if start_idx + max_len >= len(tokens) - 1:
        return -1
    else:
        for i in range(start_idx + max_len - 1, start_idx, -1):

            if tokens[i] == "\n":
                return i
            elif tokens[i] in ['.', '?', '!', '。', '！', '？']:
                return i + 1

    return -1


if __name__ == "__main__":
    # 测试数据切分代码
    file = 'data/text_score/webtext.json'
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 测试读写jsonl
    dist_file = 'data/text_score/webtext_test.jsonl'
    with open(dist_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            print(line)
            break
