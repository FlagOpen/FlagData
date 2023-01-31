# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import os
import torch
from loguru import logger
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader
from .utils.model import DistilBertModel, BertClassifier
from .utils.data import SeqCLSDataset
from .utils.datadistill import DistilledData
from .utils.common_utils import read_config, set_random_seed
from datetime import datetime
import tempfile


class DataDistillationTrainer(object):
    def __init__(self, config_path):
        self.config = read_config(config_path)
        self.model_cache_dir = tempfile.TemporaryDirectory()
        timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        logger.add(f'log/log_{timestamp}.log')
        self.data_loader = {}
        set_random_seed(self.config["basic"]["seed"])
        self.model, self.tokenizer, self.init_model_path = self.setup_model_and_tokenizer()
        self.distilled_load_path = self.config["distill"]["distilled_load_path"]
        self.distilled_save_path = self.config["distill"]["distilled_save_path"]
        self.best_loss = float("inf")

    def setup_model_and_tokenizer(self):
        distilbert_model = DistilBertModel.from_pretrained(
            self.config["model"]["model_name"])
        tokenizer = DistilBertTokenizer.from_pretrained(
            self.config["model"]["model_name"])
        model = BertClassifier(bert_model=distilbert_model,
                               num_classes=self.config["data"]["num_classes"],
                               drop_p=self.config["basic"]["drop_p"]).to(self.config["basic"]["device"])
        init_model_path = os.path.join(
            self.model_cache_dir.name, "model_init_states.pt")
        torch.save(model.state_dict(), init_model_path)
        return model, tokenizer, init_model_path

    def load_data(self, dataset_class=SeqCLSDataset):
        for type in ['train', 'test']:
            data_path = self.config["data"][f"{type}_data_path"]
            dataset = dataset_class(
                data_path, self.tokenizer, self.config["data"]["max_seq_len"])
            batch_size = self.config["data"][f"{type}_batch_size"]
            if type == "train":
                shuffle = True
            else:
                shuffle = False
            self.data_loader[type] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    def fit(self):
        distilled_data = DistilledData(
            config=self.config,
            model=self.model,
            train_loader=self.data_loader['train'],
            initial_model_path=self.init_model_path
        )
        if self.distilled_load_path is not None and os.path.exists(self.distilled_load_path):
            distilled_data.load_distilled_data(self.distilled_load_path)
        for epoch in range(self.config["distill"]["n_distill_epochs"]):
            logger.info(f"Epoch[{epoch+1}]: Train distilled data " + "-" * 70)
            distilled_data.train_distilled_data(epoch)
            logger.info(f"Epoch[{epoch+1}]: Test distilled data" + "-" * 70)
            test_loss, test_acc = distilled_data.test_distilled_data(
                self.data_loader['test'])
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                distilled_data.save_distilled_data(self.distilled_save_path)

    def __del__(self):
        self.model_cache_dir.cleanup()
