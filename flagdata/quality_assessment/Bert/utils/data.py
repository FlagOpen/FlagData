import torch
import numpy as np
import multiprocessing
import random, os, json, math
from torch.utils.data import Dataset
from utils.encode import encode_document, encode_document_with_whole_segment

asap_ranges = {
    1: (2.0, 12.0),
    2: (1.0, 6.0),
    3: (0.0, 3.0),
    4: (0.0, 3.0),
    5: (0.0, 4.0),
    6: (0.0, 4.0),
    7: (0.0, 30.0),
    8: (0.0, 60.0),
    9: (0.5, 9.0),
    10: (1.0, 24.0),
}

text_quality_ranges = {
    "pretrain": (0.0, 1.0),
    "fine_tune": (0.0, 1.0)
}

asap_essay_lengths = {
    1: 649,
    2: 704,
    3: 219,
    4: 203,
    5: 258,
    6: 289,
    7: 371,
    8: 1077,
    9: 415,
    10: 1024,
    11: 252
}


def text_quality_score(score, mode):
    """
    fix the predicted score
    """
    min_score, max_score = text_quality_ranges[mode]
    if score < min_score:
        return min_score
    elif score > max_score:
        return max_score
    else:
        return score


def fix_score(score, prompt):
    """
    fix the predicted score
    """
    if prompt == 9:  # telis
        int_part = float(int(score))
        float_part = score - int_part
        result = int_part
        if float_part < 0.25:
            result = int_part
        elif float_part < 0.75:
            result = int_part + 0.5
        else:
            result = int_part + 1

        min_score, max_score = asap_ranges[prompt]
        if result < min_score:
            return min_score
        elif result > max_score:
            return max_score
        else:
            return result

    elif prompt <= 10:
        min_score, max_score = asap_ranges[prompt]
        if score < min_score:
            return min_score
        elif score > max_score:
            return max_score
        else:
            return round(score)
    else:
        return score


def is_zh(s):
    # '包含汉字的返回TRUE'
    for c in s:
        if c >= '\u4e00' and c <= '\u9fa5':
            return True
    return False


def load_asap_data(data_file, max_len=1024, data_sample_rate=1.0):
    ids = []
    texts = []
    labels = []
    sample_index = 0
    with open(data_file) as fin:
        for line in fin:
            rand_value = random.random()
            if rand_value > data_sample_rate:
                continue
            line = line.strip()
            line_vec = line.split("\t")
            if len(line_vec) == 3:
                ids.append(line_vec[0])
                if len(line_vec[1].split(" ")) >= max_len:
                    line_vec[1] = " ".join(line_vec[1].split(" ")[0:max_len])
                texts.append(line_vec[1])
                labels.append(float(line_vec[2]))
            else:
                ids.append(str(sample_index))
                sample_index += 1
                if is_zh(line_vec[0]) and len(line_vec[0].replace(" ", "")) >= max_len:
                    line_vec[0] = line_vec[0].replace(" ", "")[0:max_len]
                elif len(line_vec[0].split(" ")) >= max_len:
                    line_vec[0] = " ".join(line_vec[0].split(" ")[0:max_len])
                texts.append(line_vec[0])
                labels.append(float(line_vec[1]))
    for id, text, label in zip(ids, texts, labels):
        yield (id, text, label)


class DocumentDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, doc_config,
                 segment_config, train_cfg):
        """
        file: 文件路径
        tokenizer: 分词器
        max_len: 最大长度
        doc_config: Doc Squences
        segment_config: {"Segment Scale": float, "Segment Squences": float}
        train_cfg: 训练配置
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.segment_scales = segment_config["scales"]
        self.segment_sequences = segment_config["sequences"]
        self.doc_sequences = doc_config["sequences"]
        self.whole_segment = train_cfg.whole_segment

        # 文件加载
        if file.endswith(".json"):
            # 加载json格式文件
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 提取 text 与 score 字段
                self.data = [{"text": item["text"],
                              "score": 1 if item["score"] == 1 else train_cfg.cc_score
                              } for item in data]

        elif file.endswith(".jsonl"):
            # 加载jsonl格式文件
            self.data = []
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    self.data.append(
                        {"text": item["text"],
                         "score": 1 if item["score"] == 1 else train_cfg.cc_score})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]
        score = item["score"]

        # BERT-DOC-TOKEN 的输入
        if self.whole_segment:
            _, representation_doc_token = encode_document_with_whole_segment(
                text, self.tokenizer, self.max_len)
            representation_doc_token = representation_doc_token[0, :, :].unsqueeze(0)

        else:
            representation_doc_token = encode_document(text, self.tokenizer,
                                                       self.max_len, self.doc_sequences)

        label = torch.tensor(score, dtype=torch.float)

        return representation_doc_token, label


class DocumentDatasetSiamese(DocumentDataset):
    def __init__(self, file, tokenizer, max_len, doc_config, segment_config, train_cfg):
        super().__init__(file, tokenizer, max_len, doc_config, segment_config, train_cfg)
        self.cc_score = train_cfg.cc_score
        self.file = file

        data_num = len(self.data) - len(self.data) % train_cfg.batch_size
        self.data = self.data[:data_num]

        # 每个样本设一个唯一的id
        for index, item in enumerate(self.data):
            item["id"] = index

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]
        score = item["score"]

        # BERT-DOC-TOKEN 的输入
        _, representation_doc_token = encode_document_with_whole_segment(
            text, self.tokenizer, self.max_len)
        representation_doc_token = representation_doc_token[0, :, :].unsqueeze(0)
        label = torch.tensor(score, dtype=torch.float)

        return torch.tensor(item["id"], dtype=torch.int), representation_doc_token, label

    def get_length(self):
        return len(self.data)

    def data_revision(self, false_positive_ids, false_negative_ids):
        # 将部分负样本更改为正样本(score = 1)
        for id in false_negative_ids:
            self.data[id]["score"] = 1

        # 将部分正样本更改为负样本(score = 0)
        for id in false_positive_ids:
            self.data[id]["score"] = self.cc_score


class DocumentDatasetForPredict(Dataset):
    def __init__(self, file, tokenizer, max_len, doc_config,
                 segment_config, train_cfg):
        """
        file: 文件路径
        tokenizer: 分词器
        max_len: 最大长度
        doc_config: Doc Squences
        segment_config: {"Segment Scale": float, "Segment Squences": float}
        train_cfg: 训练配置
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.segment_scales = segment_config["scales"]
        self.segment_sequences = segment_config["sequences"]
        self.doc_sequences = doc_config["sequences"]

        key = train_cfg.text_key

        # 文件加载
        if file.endswith(".json"):
            # 加载json格式文件
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 提取 text 与 score 字段
                self.data = [{"text": item[key],
                              "url": item["url"],
                              "title": item['title'],
                              "source_domain": item['source_domain']
                              } for item in data]


        elif file.endswith(".jsonl"):
            # 加载jsonl格式文件
            self.data = []
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    if len(item[key].strip()) > 0:
                        self.data.append(
                            {"text": item[key],
                             "url": item["url"],
                             "title": item['title'],
                             "source_domain": item['source_domain']})

    def __len__(self):
        print("=================>")
        print(self)
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]
        info = {"url": item["url"], "title": item["title"], "source_domain": item["source_domain"]}

        text_segments, representation_doc_token = encode_document_with_whole_segment(
            text, self.tokenizer, self.max_len)

        text_index = [index] * len(text_segments)

        return text_index, text_segments, representation_doc_token, info

    def get_length(self):
        return len(self.data)


class DocumentDatasetForEvaluation(DocumentDataset):
    def __init__(self, file, tokenizer, max_len, doc_config, segment_config, train_cfg):
        super().__init__(file, tokenizer, max_len, doc_config, segment_config, train_cfg)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]
        score = item["score"]

        # BERT-DOC-TOKEN 的输入
        representation_doc_token = encode_document(text, self.tokenizer,
                                                   self.max_len, self.doc_sequences)

        # BERT-Segment 的输入 
        representation_segment_list = []
        for i in range(len(self.segment_scales)):
            representation_segment = encode_document(text, self.tokenizer,
                                                     self.segment_scales[i],
                                                     self.segment_sequences[i])
            representation_segment_list.append(representation_segment)

        label = torch.tensor(score, dtype=torch.float)

        return ([text], representation_doc_token, representation_segment_list), label


class DocumentDatasetForUST(Dataset):

    def __init__(self, train_cfg, tokenizer, max_len):
        """
        训练数据采集方式
        ust 采用ust半监督训练的方法，正样本是模型给出的        
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_cfg = train_cfg
        self.train_folder = train_cfg.train_folder
        self.file_list = [file for file in os.listdir(self.train_folder) if file.endswith(".json")]
        self.mode = "sample"
        self.data = None

        if train_cfg.use_paradigm:
            self.paradigm_data_file = train_cfg.paradigm_data_file
            self.paradigm_data_generate()

    def paradigm_data_generate(self):
        self.paradigm_texts = []
        with open(self.paradigm_data_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            for item in json_data:
                if item["score"] == 1:
                    self.paradigm_texts.append(item["text"])

    def candidate_data_generate(self):
        selected_file_list = random.sample(self.file_list, self.train_cfg.select_file_num)
        total_num = int(self.train_cfg.per_jsonfile_text_num * len(selected_file_list))
        # 生成随机 candidate_data_num 个 index
        random_index = random.sample(range(total_num), int(self.train_cfg.candidate_data_num))

        index_for_file = {}

        # 根据文件拆分index
        for n in random_index:
            quotient = int(n // self.train_cfg.per_jsonfile_text_num)
            remainder = int(n % self.train_cfg.per_jsonfile_text_num)
            if selected_file_list[quotient] not in index_for_file:
                index_for_file[selected_file_list[quotient]] = []
            index_for_file[selected_file_list[quotient]].append(remainder)

        # 多进程读取数据
        max_process = self.train_cfg.num_workers
        pool = multiprocessing.Pool(processes=max_process)
        results = []
        for file, indexes in index_for_file.items():
            result = pool.apply_async(self.collect_data, (self.train_folder + file, indexes))
            results.append(result)

        pool.close()
        pool.join()

        self.data = []
        for result in results:
            self.data.extend(result.get())

        # 给每个text一个唯一的id
        for index, item in enumerate(self.data):
            item["id"] = index

    def select_samples(self, ids: list, pseudo_labels: list, confidences: list):
        """
        indexes: 被选择的，用于训练的样本，每个index在 [0, candidate_data_num - 1]之间
        """
        assert len(ids) == len(pseudo_labels)
        input_dict = {ids[n]: {"pseudo_label": pseudo_labels[n], "confidence": confidences[n]
                               } for n in range(len(ids))}

        dict_with_id_key = {item["id"]: item for item in self.data}
        self.data = [{"text": dict_with_id_key[identity]["text"],
                      "pseudo_label": input_dict[identity]["pseudo_label"],
                      "confidence": input_dict[identity]["confidence"]
                      } for identity in ids]

        if self.train_cfg.use_paradigm:
            paradigm_confidence = np.mean(np.array([self.data[n][
                                                        "confidence"] for n in range((len(self.data)))]))

            neg_data = [item for item in self.data if int(item['pseudo_label']) == 0]

            if self.train_cfg.data_scheme == "paradigm":
                # 全部正样本为范例样本
                self.data = neg_data
                indexes = random.sample(range(len(self.paradigm_texts)), len(self.data))

            elif self.train_cfg.data_scheme == "ust_paradigm":
                # 一定比例的样本为范例样本，另一部分为Common Crawl中判为正样本的，抽样得到的结果
                paradigm_num = int(self.train_cfg.paradigm_ratio * len(neg_data))
                cc_pos_num = len(neg_data) - paradigm_num

                pos_data = [item for item in self.data if int(item['pseudo_label']) == 1]
                indexes = random.sample(range(len(pos_data)), cc_pos_num)
                pos_data = [pos_data[n] for n in indexes]
                self.data = neg_data + pos_data
                indexes = random.sample(range(len(self.paradigm_texts)), paradigm_num)

            select_paradigm_data = [{
                "text": self.paradigm_texts[index], "pseudo_label": 1.0,
                "confidence": paradigm_confidence
            } for index in indexes]
            self.data.extend(select_paradigm_data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item["text"]

        # BERT-DOC-TOKEN 的输入
        text_segs, representation_doc_token = encode_document_with_whole_segment(
            text, self.tokenizer, self.max_len)
        representation_doc_token = representation_doc_token[0, :, :].unsqueeze(0)

        if self.mode == "sample":
            return torch.tensor(item["id"], dtype=torch.int), representation_doc_token, text_segs[0]
        elif self.mode in ["train", "eval"]:
            label = torch.tensor(item["pseudo_label"], dtype=torch.float)
            confidence = torch.tensor(item["confidence"], dtype=torch.float)
            return representation_doc_token, label, confidence, text_segs[0]

    @staticmethod
    def collect_data(file, indexes):
        data = []

        with open(file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            for index in indexes:
                data.append({'text': json_data[index]['raw_content']})
        # print(file)
        return data

    def set_mode(self, mode="sample"):
        """
        数据集是用于MC Dropout采样 or 训练
        mode: str "sample", "train" 或 "eval"
        """
        self.mode = mode

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import time
    from transformers import BertTokenizer, BertConfig
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from types import SimpleNamespace

    # # 测试数据加载器
    # with open("configs/base_config.json", "r", encoding="utf-8") as f:
    #     train_cfg = json.load(f)
    # train_cfg = SimpleNamespace(**train_cfg)
    # model_cfg = BertConfig.from_pretrained(train_cfg.bert_model_path)
    # tokenizer = BertTokenizer.from_pretrained(train_cfg.tokenizer_path)
    # max_len = 512
    # doc_cfg = {"sequences": 1}
    # segment_cfg = {"scales": [90, 30, 130, 10], 
    #                "sequences": [12, 36, 9, 108]}

    # # 测试Dataset输出shape不一致的情况
    # dataset = DocumentDatasetForPredict(train_cfg.train_file, tokenizer, 
    #                                 model_cfg.max_position_embeddings, 
    #                                 model_cfg.doc_cfg, model_cfg.segment_cfg, 
    #                                 train_cfg)

    # dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
    # epoch = 0
    # num_epochs = 100

    # text_index_list = []
    # text_segments_list = []
    # representation_segment_list = []
    # for step, (text_index, text_segs, representation) in enumerate(
    #     tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):

    #     text_index_list.extend(text_index)
    #     text_segments_list.extend(text_segs)
    #     representation_segment_list.append(representation)

    #     if len(text_index_list) >= 64:
    #         representation_doc_token = torch.cat(representation_segment_list, dim=0)
    #     print("success")

    # 测试DocumentDatasetForUST数据集
    with open("configs/ust_config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)
    train_cfg = SimpleNamespace(**train_cfg)
    model_cfg = BertConfig.from_pretrained(train_cfg.bert_model_path)
    tokenizer = BertTokenizer.from_pretrained(train_cfg.tokenizer_path)
    max_len = model_cfg.max_position_embeddings
    start_time = time.time()
    dataset = DocumentDatasetForUST(train_cfg, tokenizer, max_len)
    end_time = time.time()
    print(end_time - start_time)
    data = dataset.data
    print(len(data))
