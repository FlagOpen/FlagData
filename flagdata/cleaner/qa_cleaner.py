from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import json
import jieba_fast as jieba
import logging
from tqdm import tqdm
import warnings
import pdb


class QACleaner():
    def __init__(self):
        super().__init__()
        '''
        Using a trained classfier to filter the QA texts using labeled instances.
        The labeled instances for training the classfier is the self.train
        '''
        train_dat = self.load_jsonl(
            'input/ref.jsonl')
        self.train = [sample for sample in train_dat if sample['label'] in ['0', '1']]
        if len(self.train) != train_dat:
            warnings.warn('Training data contains label beyond "0" and "1"!', Warning)

    def clean(self):
        print("qa_cleaner start......")
        text_train = [sample['text'] for sample in self.train]
        label_train = [sample['label'] for sample in self.train]

        for text in tqdm(self.read_jsonl_in_chunks(
                'input/qa_demo_input.jsonl',
                300000)):
            text = [i['content'] for i in text]
            filtered_text, discard_text = self.filter(text_train, label_train, text)
            self.write_jsonl_file(filtered_text,
                                  'output/qa_demo_output_filtered.jsonl')
            self.write_jsonl_file(discard_text,
                                  'output/qa_demo_output_discarded.jsonl')

    def filter(self, text_train, labels_train, text_test):
        num_train = len(text_train)

        # tokenize training and test text
        print('Tokenizing input text')
        tokenized_text_train = [list(set(jieba.lcut(text))) for text in text_train]
        tokenized_text_test = [list(set(jieba.lcut(text))) for text in text_test]
        tokenized_text = tokenized_text_train + tokenized_text_test
        # One-Hot encoding
        mlb = MultiLabelBinarizer(sparse_output=True)  # store using sparse matrix
        one_hot_encoded_data = mlb.fit_transform(tokenized_text)
        one_hot_encoded_data_train = one_hot_encoded_data[:num_train, ]
        one_hot_encoded_data_test = one_hot_encoded_data[num_train:, ]

        # Fitting Ridge logistic regression model
        print(f'Fitting Ridge logistic regression model with alpha={str(150)}')
        model = RidgeClassifier(alpha=150)
        model.fit(one_hot_encoded_data_train, labels_train)

        decision_function = model.decision_function(one_hot_encoded_data_test)
        probabilities = 1 / (1 + np.exp(-decision_function))

        # predicting label using this model
        print('Predicting label')
        mlb = MultiLabelBinarizer(sparse_output=True)  # 使用稀疏矩阵来存储
        predictions = model.predict(one_hot_encoded_data_test)

        dat_keep = [{'text': text_test[i], 'meta': {"prob": probabilities[i]}} for i in
                    range(len(text_test)) if predictions[i] == '0']
        dat_discard = [{'text': text_test[i], 'meta': {"prob": probabilities[i]}} for i in
                       range(len(text_test)) if predictions[i] == '1']
        return dat_keep, dat_discard

    @staticmethod
    def load_jsonl(path):
        dat = open(path, 'r').readlines()
        dat = [json.loads(i) for i in dat]
        return dat

    @staticmethod
    def read_jsonl_in_chunks(path, batch_size):
        """逐块读取 JSONL 文件。
        Args:
            file_path (str): JSONL 文件路径。
            chunk_size (int): 每个块包含的行数。

        Yields:
            list: 包含多个 JSON 对象的列表。
        """
        with open(path, 'r', encoding='utf-8') as file:
            chunk = []
            for line in file:
                try:
                    chunk.append(json.loads(line.strip()))  # 解析 JSON 行
                except Exception as e:
                    logging.warning("read error: ", e)
                if len(chunk) == batch_size:
                    yield chunk
                    chunk = []
            if chunk:  # 处理剩余的部分
                yield chunk

    @staticmethod
    def write_jsonl_file(sample_ls, path):
        with open(path, 'a+', encoding='utf-8') as f:
            for ipt in sample_ls:
                json_str = json.dumps(ipt, ensure_ascii=False)
                f.write(json_str + '\n')


if __name__ == '__main__':
    qa_cleaner = QACleaner()
    qa_cleaner.clean()
