# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import json
import torch
from torch.utils.data import Dataset
import pandas as pd


class SeqCLSDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = self.read_data(data_path)
        self.sentence1 = self.data['text'].values
        self.targets = self.data['label'].values
        self.max_length = max_length

    def read_data(self, path):
        data, label = [], []
        with open(path, "r", encoding="utf8") as f:
            for line in f.readlines():
                line = json.loads(line)
                data.append(line['text'])
                label.append(line['label'] - 1)
        return pd.DataFrame(data={'text': data, 'label': label})

    def __len__(self):
        return len(self.sentence1)

    def __getitem__(self, index):
        sent1 = str(self.sentence1[index])

        inputs = self.tokenizer.encode_plus(
            sent1,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt", truncation=True
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return input_ids.squeeze(), attention_mask.squeeze(), torch.tensor(self.targets[index])
