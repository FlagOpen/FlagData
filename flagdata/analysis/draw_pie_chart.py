# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import matplotlib.pyplot as plt

# 给定的数据列表
x = ['people', 'geography', 'history', 'science and technology', 'culture', 'nature', 'sports', 'economy', 'politics',
     'society', 'military category', 'Religion', 'Health', 'Education', 'Law', 'Entertainment', 'Food', 'Tourism',
     'Agriculture', 'Environment']
y = [2.97, 13.51, 3.42, 24.84, 10.26, 4.12, 3.75, 7.40, 4.52, 1.41, 0.82, 0.65, 2.97, 3.86, 1.66, 7.66, 3.93, 0.95,
     1.03, 0.28]

# 创建饼图
plt.figure(figsize=(8, 8))  # 设置图表大小
plt.pie(y, labels=x, autopct='%1.1f%%', startangle=140)
plt.title('Pie chart of each category')
plt.axis('equal')  # 使饼图保持圆形

# 保存饼图为PNG文件
plt.savefig('pie_chart.png', bbox_inches='tight')  # 将文件保存为名为 "pie_chart.png" 的PNG格式文件
plt.show()
