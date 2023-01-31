# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import random
import torch
import numpy as np
import pathlib
import yaml


def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()


def read_config(config_path):
    with open(config_path, "r", encoding="utf8") as fr:
        config = yaml.safe_load(fr)
    device_id = config["basic"]["cuda_device_id"]
    if device_id != -1:
        config["basic"]["device"] = torch.device(f"cuda:{device_id}")
    else:
        config["basic"]["device"] = torch.device("cpu")
    config["basic"]["dtype"] = torch.bfloat16 if config["basic"]["fp16"] else torch.float16
    return config
