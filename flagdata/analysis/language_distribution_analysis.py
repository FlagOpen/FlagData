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

# 为展示效果，使用如下数据，具体使用时，需要用languages、counts替换labels和values_pie、values_bar
from plotly.subplots import make_subplots
import plotly.graph_objects as go

labels = ["英语", "俄语", "德语", "西班牙语", "法语", "日语", "葡萄牙语", "意大利语",
          "波斯语", "波兰语", "中文", "荷兰语", "土耳其语", "捷克", "朝鲜", "越南",
          "阿拉伯", "希腊语", "瑞典", "罗马尼亚", "斯洛伐克", "丹麦", "印度尼西亚",
          "芬兰", "泰国", "保加利亚语", "乌克兰", "希伯来语", "挪威语", "克罗地亚",
          "立陶宛", "塞尔维亚", "挪威", "斯洛文尼亚", "巴伦西亚语", "拉脱维亚",
          "爱沙尼亚语", "印地语"]
values_pie = [54.1, 6.0, 6.0, 4.9, 4.0, 3.4, 2.9, 2.3, 2.0, 1.7, 1.7, 1.2, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4,
              0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
values_bar = [54.1, 6.0, 6.0, 4.9, 4.0, 3.4, 2.9, 2.3, 2.0, 1.7, 1.7, 1.2, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5, 0.5, 0.4, 0.4,
              0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Create a subplot with two columns
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'bar'}]])

# Add pie chart to the first column
fig.add_trace(go.Pie(labels=labels, values=values_pie, name="Language Distribution"), row=1, col=1)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title_text="Language Distribution")

# Add bar chart to the second column
fig.add_trace(go.Bar(x=labels, y=values_bar), row=1, col=2)

# Show the combined figure
fig.show()
