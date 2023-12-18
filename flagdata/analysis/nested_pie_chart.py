# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")


import matplotlib.pyplot as plt
import numpy as np

# 定义你提供的数据
data = {
    "Understanding ability": {"Total_number": 40, "Secondary_classification": {
        "Information Analysis": 10,
        "Information extraction": 10,
        "Information Summary": 10,
        "Cross linguistic understanding": 5,
        "Discriminant evaluation": 5
    }},
    "Knowledge application": {"Total_number": 20, "Secondary_classification": {
        "Knowledge Q&A": 10,
        "Common sense Q&A": 5,
        "Fact Q&A": 5
    }},
    "Inference ability": {"Total_number": 10, "Secondary_classification": {
        "Knowledge reasoning": 5,
        "Symbolic reasoning": 5
    }},
    "Generative ability": {"Total_number": 7, "Secondary_classification": {
        "Conditional generation": 4,
        "Code generation": 2,
        "Creative Generation": 1
    }},
    "Explanatory ability": {"Total_number": 5, "Secondary_classification": {
        "Psychological counseling": 3,
        "Language Analysis": 2
    }},
    "Values": {"Total_number": 5, "Secondary_classification": {
        "Discrimination and Prejudice": 2,
        "Security": 2,
        "Ethics and Morality": 1
    }},
    "Cultural understanding": {"Total_number": 7, "Secondary_classification": {
    }},
    "Illusion": {"Total_number": 3, "Secondary_classification": {
    }},
    "General comprehensive ability": {"Total_number": 2, "Secondary_classification": {
    }},
    "Domain expert capabilities": {"Total_number": 1, "Secondary_classification": {
    }}
}

# 准备绘图数据
categories = list(data.keys())
outer_labels = []
outer_sizes = []
inner_labels = []
inner_sizes = []

# 遍历一级分类
for category in categories:
    outer_labels.append(category)
    outer_sizes.append(data[category]["Total_number"])

    # 获取二级分类数据
    subcategories = list(data[category]["Secondary_classification"].keys())
    inner_labels.extend(subcategories)
    inner_sizes.extend(list(data[category]["Secondary_classification"].values()))

# 绘制双层环形图
fig, ax = plt.subplots()
ax.axis('equal')

# 外层环形图
wedges1 = ax.pie(outer_sizes, labels=outer_labels, radius=1, autopct='%1.1f%%', startangle=90)

# 内层环形图
wedges2 = ax.pie(inner_sizes, radius=0.7, startangle=90)[0]  # 获取饼图块列表

# 设置内层环形图标签
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges2):
    ang = (p.theta2 - p.theta1) / 2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(inner_labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                horizontalalignment=horizontalalignment, **kw)

plt.show()
