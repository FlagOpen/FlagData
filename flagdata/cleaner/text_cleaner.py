# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from tqdm import tqdm
from .utils.filter import CleanerPipeline
from .utils.extractor import ExtractorPipeline
from .utils.initializer import ProcessorInitializer
from .utils.loggings import count_global, count_positive, err_callback
import re
import os
import logging
import json
import yaml

LOGLEVEL = os.environ.get('LOGLEVEL')
if LOGLEVEL:
    LOGLEVEL = LOGLEVEL.upper()
else:
    LOGLEVEL = "INFO"
logging.basicConfig(level=LOGLEVEL)


class DataCleaner:
    """
    text cleaner
    """
    def __init__(self, config_path="config.yaml"):
        self.config = self._read_config(config_path)
        logging.info(self.config)

    @staticmethod
    def _read_config(config_path: str):
        with open(config_path, "r", encoding="utf8") as fr:
            return yaml.safe_load(fr)

    def read_data(self, smm):
        """
        read input data
        """
        data = []
        batch = []
        num_lines = 0
        input_path = self.config["basic"].get("input")
        batch_size = self.config["basic"].get("batch_size")
        input_files = [input_path] if not os.path.isdir(input_path) \
            else [os.path.join(input_path, filename) for filename in os.listdir(input_path)]
        for full_path in input_files:
            with open(full_path, "r", encoding="utf8") as fr:
                for line in fr:
                    line = line.strip()
                    batch.append(line.encode("utf8"))
                    num_lines += 1
                    if num_lines % batch_size == 0:
                        data.append(smm.ShareableList(batch))
                        batch = []
                    if num_lines % 500000 == 0:
                        logging.info(f"reading {num_lines}...")
                        print(line)
        if len(batch):
            data.append(smm.ShareableList(batch))
        return data, num_lines

    def write2file(self, cleaned_data, num_batches):
        """
        write to file
        """
        with open(self.config["basic"].get("output"), "w", encoding="utf8") as fw:
            for batch_idx in tqdm(range(num_batches)):
                try:
                    cleaned_batch = cleaned_data[batch_idx]
                    for line in cleaned_batch:
                        fw.write(json.dumps(line, ensure_ascii=False) + "\n")
                except Exception as e:
                    logging.warning("SKIP WRITING BATCH: | ", e, batch_idx)

    def clean_batch(self, batch, batch_idx, num_batches, is_jsonl: bool, source_key: str, result_key: str):
        """
        clean batch data
        """
        cleaned_batch = []
        for line in batch:
            if not is_jsonl:
                # mode: plain text
                source_key = "rawContent"
                if not result_key:
                    result_key = "cleanedContent"
                data_i = {source_key: line.decode("utf8")}
            else:
                # mode: jsonl
                data_i = json.loads(line.decode("utf8"))
            try:
                content = data_i[source_key]
                if extractor is not None:
                    extracted_items, content = extractor.extract(content)
                    data_i.update(extracted_items)
                if cleaner is not None:
                    cleaned_content = cleaner.clean(content)
                    # simple post-processing for deplicated white spaces and line-break
                    cleaned_content = re.sub(
                        "\s+", " ", cleaned_content).strip()
                    cleaned_content = re.sub("[\r\n]+", "\n", cleaned_content)
                    data_i[result_key] = cleaned_content
            except Exception as e:
                logging.warning(f"SKIP ERROR LINE: | {line}")
                logging.warning(f"INFO: {e}")
                continue
            if content != cleaned_content:
                count_positive(shared_symbol_counter, content, cleaned_content)
            cleaned_batch.append(data_i)
        shared_cleaned_data[batch_idx] = cleaned_batch

        count_global(shared_global_counter, shared_s, num_batches)

    def init_pool(self, symbol_counter, global_counter, cleaned_data, s):
        global shared_symbol_counter, shared_global_counter, \
            shared_cleaned_data, shared_s, extractor, cleaner
        shared_symbol_counter = symbol_counter
        shared_global_counter = global_counter
        shared_s = s
        shared_cleaned_data = cleaned_data

        extractor_initializer = ProcessorInitializer(self.config, "extractors")
        filter_initializer = ProcessorInitializer(self.config, "filters")
        assert extractor_initializer.num_processors + filter_initializer.num_processors > 0, \
            "At least one processor need to be activate in config file"

        extractors = [extractor for extractor in extractor_initializer]
        if extractor_initializer.num_processors > 0:
            extractor = ExtractorPipeline(*extractors)
        else:
            extractor = None

        filters = [filter for filter in filter_initializer]
        if filter_initializer.num_processors > 0:
            cleaner = CleanerPipeline(*filters)
        else:
            cleaner = None

    def clean(self):
        """
        clean data with multiprocessing and shared memory support
        """
        shared_memory_manager = SharedMemoryManager()
        mp_manager = mp.Manager()
        shared_memory_manager.start()
        shared_data, num_lines = self.read_data(shared_memory_manager)
        symbol_counter = mp.Value("i", 0)
        global_counter = mp.Value("i", 0)

        num_batches = len(shared_data)
        cleaned_data = mp_manager.dict({k: None for k in range(num_batches)})

        logging.info(f"read data finished, #total lines: {num_lines}")
        s = time.perf_counter()
        pool = mp.Pool(self.config["basic"].get("num_workers"), initializer=self.init_pool, initargs=(
            symbol_counter, global_counter, cleaned_data, s))
        source_key = self.config["basic"].get("source_key")
        result_key = self.config["basic"].get("result_key")
        is_jsonl = self.config["basic"].get("is_jsonl")
        for batch_idx, batch in enumerate(shared_data):
            batch_args = (batch,
                          batch_idx,
                          num_batches,
                          is_jsonl,
                          source_key,
                          result_key
                          )
            pool.apply_async(self.clean_batch, args=batch_args,
                             error_callback=err_callback)
        pool.close()
        pool.join()
        e = time.perf_counter()
        logging.info(
            f"|cost {e - s} seconds! | cleaned {num_batches} / {num_batches} batches......")
        logging.info(f"#positive cleaned: {symbol_counter.value}")
        logging.info(f"write to file {self.config['basic'].get('output')}")
        self.write2file(cleaned_data, num_batches)
        shared_memory_manager.shutdown()
