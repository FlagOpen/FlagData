# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import jsonlines
import statistics

file_path = './data/language_distribution_data.jsonl'  # JSONL文件路径

total_texts = []  # 存储所有文本

with jsonlines.open(file_path) as reader:
    for line in reader:
        text = line['text']  # 假设JSONL中的文本字段名为'text'
        total_texts.append(text)

# 计算所有文本的长度统计信息
total_lengths = [len(text) for text in total_texts]
average_length = sum(total_lengths) / len(total_lengths)
max_length = max(total_lengths)
min_length = min(total_lengths)
median_length = statistics.median(total_lengths)

print(f"Average Length of all texts: {average_length}")
print(f"Maximum Length of all texts: {max_length}")
print(f"Minimum Length of all texts: {min_length}")
print(f"Median Length of all texts: {median_length}")
