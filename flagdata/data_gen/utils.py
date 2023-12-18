# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import json
import openai


def load_jsonl(file_path):
    dat = open(file_path, 'r').readlines()
    dat = [json.loads(i) for i in dat]
    return dat


def save_jsonl(sample_ls, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for ipt in sample_ls:
            json_str = json.dumps(ipt, ensure_ascii=False)
            f.write(json_str + '\n')


def clean_patterns(txt, patterns):
    for pattern in patterns:
        txt = txt.replace(pattern, '')
    return txt


def get_gpt4_response(query, deployment_name='gpt4'):
    response = openai.ChatCompletion.create(
        engine=deployment_name,  # engine = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    return response['choices'][0]['message']['content']
