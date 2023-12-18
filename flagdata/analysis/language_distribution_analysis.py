# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import jsonlines
from pathlib import Path

from flagdata.language_identification.split_by_lang import Classifier

file_path = './data/language_distribution_data.jsonl'  # JSONL文件路径

languages_count = {}  # 用于存储每种语言的出现次数

model = Path(
    "../language_identification/bin/lid.bin")

classifier = Classifier(model, "text", "lang")
classifier.__enter__()
with jsonlines.open(file_path) as reader:
    for line in reader:
        text = line['text']  # 假设JSONL中的文本字段名为'text'
        try:
            language = classifier(dict(text=text)).get('lang')  # 使用语言检测库进行语言识别
            if language in languages_count:
                languages_count[language] += 1
            else:
                languages_count[language] = 1
        except:
            pass  # 忽略无法识别语言的文本行

# 输出每种语言的出现次数
for lang, count in languages_count.items():
    print(f"Language: {lang}, Count: {count}")

import matplotlib.pyplot as plt

# 将语言和对应的频率数据转换为列表，方便绘图
languages = list(languages_count.keys())
counts = list(languages_count.values())

# plt.figure(figsize=(8, 6))
# plt.bar(languages, counts)
# plt.xlabel('Languages')
# plt.ylabel('Frequency')
# plt.title('Language Distribution in the File')
# plt.xticks(rotation=45)  # 旋转x轴标签，以避免重叠
# plt.tight_layout()
# plt.show()

# 绘制饼图
plt.figure(figsize=(8, 6))
plt.pie(counts, labels=languages, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # 使饼图比例相等
plt.title('Language Distribution')

plt.tight_layout()
plt.show()
