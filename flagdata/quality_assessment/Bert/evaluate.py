"""
@misc{chen2023chinesewebtext,
      title={ChineseWebText: Large-scale High-quality Chinese Web Text Extracted with Effective Evaluation Model},
      author={Jianghao Chen and Pu Jian and Tengxiao Xi and Dongyi Yi and Qianlong Du and Chenglin Ding and Guibo Zhu and Chengqing Zong and Jinqiao Wang and Jiajun Zhang},
      year={2023},
      eprint={2311.01149},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

import numpy as np
import logging
from tqdm import tqdm
import torch, json, os
from types import SimpleNamespace
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from utils.data import DocumentDatasetForPredict
from network.model_architechure_bert_multi_scale import DocumentBertScoringModel


def text_select_with_pred(file, pred_list, text_index_list,
                          text_segments_list, info_list, config):
    # Write file address
    file_name = os.path.basename(file)
    filtered_file_name = config.output_path + "/" + os.path.basename(file).replace(".jsonl", "_filtered.jsonl")
    dist_file = config.output_path + "/" + file_name

    # Determine the text to retain
    text_id_list = list(set(text_index_list))
    text_index_array = np.array(text_index_list)
    pred_bool_array = np.array(pred_list) > config.score_threshold

    for text_id in text_id_list:

        indexes = np.where(text_index_array == text_id)[0]
        cur_text_bool = pred_bool_array[indexes]

        # Object that is continuously determined to correspond to high-quality text index
        continuous_ranges = np.where(np.diff(np.concatenate((
            [False], cur_text_bool, [False]))))[0].reshape(-1, 2)

        if len(continuous_ranges) == 0:
            text = "".join([text_segments_list[index] for index in indexes])
            score = np.mean(np.array(pred_list)[np.array(indexes)])
            text_dict = {"text_id": text_id, "text": text, "info": info_list[
                text_id_list.index(text_id)], "score": score}

            # Write text that is not in a contiguous interval to a file (jsonl format)
            with open(filtered_file_name, "a", encoding="utf-8") as f:
                json.dump(text_dict, f, ensure_ascii=False)
                f.write('\n')

            continue

        index_lists = []
        for start, end in continuous_ranges:
            index_list = [index for index in indexes[start:end]]
            index_lists.append(index_list)
            text = "".join([text_segments_list[index] for index in index_list])
            score = np.mean(np.array(pred_list)[np.array(index_list)])
            text_dict = {"text_id": text_id, "text": text, "info": info_list[
                text_id_list.index(text_id)], "score": score}

            # Writes filtered text to a file (jsonl format)
            with open(dist_file, "a", encoding="utf-8") as f:
                json.dump(text_dict, f, ensure_ascii=False)
                f.write('\n')

        # Write text that is not in a contiguous interval to a file (jsonl format)
        for index in indexes:
            flag = False
            # Determine whether it is in a continuous interval
            for index_list in index_lists:
                if index in index_list:
                    flag = True
                    break

            if not flag:
                score = pred_list[index]
                text_dict = {"text_id": text_id, "text": text_segments_list[index], "info": info_list[
                    text_id_list.index(text_id)], "score": score}

                with open(filtered_file_name, "a", encoding="utf-8") as f:
                    json.dump(text_dict, f, ensure_ascii=False)
                    f.write('\n')


def predict(file, model: torch.nn.Module, tokenizer: BertTokenizer,
            config: SimpleNamespace):
    # Create data_loader
    model_cfg = BertConfig.from_pretrained(config.bert_model_path)
    num_workers = config.num_workers
    dataset = DocumentDatasetForPredict(file, tokenizer,
                                        model_cfg.max_position_embeddings,
                                        model_cfg.doc_cfg, model_cfg.segment_cfg,
                                        config)

    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        shuffle=False)

    # Forecast
    world_size = torch.cuda.device_count()

    model.eval()

    text_index_list = []
    text_segments_list = []
    representation_segment_list = []
    info_list = []
    with torch.no_grad():
        for text_index, text_segments, representation, info in tqdm(
                dataloader, desc=f"Evaluation", leave=False):

            text_index_list.extend(text_index)
            text_segments_list.extend(text_segments)
            representation_segment_list.append(representation)
            info_list.append(info)

            if len(text_index_list) < world_size * config.batch_size:
                continue

            # inference
            try:
                # Splicing segments with torch
                representation_doc_token = torch.cat(representation_segment_list, dim=0)
            except Exception as e:
                logging.warning(f"representation_segment_list ERROR :  {representation_segment_list}")
                logging.warning(f"INFO: {e}")
                continue
            # representation_doc_token = torch.cat(representation_segment_list, dim=0)
            # (batch_size, seq_len, hidden_size) -> (batch_size, 1, seq_len, hidden_size)
            representation_doc_token = representation_doc_token.unsqueeze(dim=1)
            with torch.cuda.amp.autocast():
                pred = model(representation_doc_token)

            # Integrate retained text based on text index, text segments, pred
            text_select_with_pred(file, pred.tolist(), text_index_list,
                                  text_segments_list, info_list, config)


def predict_setup(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define the model (data parallelism)
    model = DocumentBertScoringModel(config)
    checkpoint = torch.load(config.checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model = torch.nn.DataParallel(model)

    # dataloader
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_path)

    return model, tokenizer, device


def process(config):
    model, tokenizer, device = predict_setup(config)

    # Get the json files under data_path
    data_path = config.data_path
    json_files = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    print(json_files)

    for file in json_files:
        print("Start reasoning, file: " + file)
        predict(file, model, tokenizer, config)


import yaml
from types import SimpleNamespace

if __name__ == "__main__":
    # Load the configuration from the YAML file
    with open("bert_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    # Call the process function and pass the configuration
    process(cfg)
