# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import jsonlines

file_path = './data/average_rounds_data.jsonl'  # JSONL文件路径

total_texts = []  # 存储所有文本

with jsonlines.open(file_path) as reader:
    for line in reader:
        text = line['text']  # 假设JSONL中的文本字段名为'text'
        total_texts.append(text)

# 计算平均轮次（以换行符为例）
total_rounds = sum(text.count('\n') for text in total_texts) / len(total_texts)

print(f"Average Rounds in texts: {total_rounds}")
