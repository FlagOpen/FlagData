# 数据预处理阶段>质量评估
选择BERT和fasttext作为评估模型，是因为它们具有以下优点：

1. BERT模型在文本分类和理解任务中表现出色，具有强大的语言理解和表示能力，能够有效地评估文本质量。
2. FastText模型具有高效的训练和推理速度，同时保持分类性能，可以显著减少训练和推理时间，fasttext的版本号0.9.2

文章比较了不同的文本分类模型，包括逻辑回归、BERT和FastText，以评估它们的性能。在实验中，BERTEval和FastText模型在文本分类任务中表现良好，其中FastText模型在精度和召回率方面表现最佳。【实验结果来自ChineseWebText】

![Classification results of different evaluation models.](quality_assessment.png)
Bert执行 python Bert/evaluate.py
示例数据为 `flagdata/quality_assessment/Bert/input_data/example_data.jsonl`，执行后的结果`cleared*.jsonl` 示例如下

```json
{
  "text_id": 1,
  "text": "在强调本金安全的前提下，追求较高的当期收入和总回报。\r\n    投资策略\t本基金将在遵守投资纪律并有效管理风险的基础上，通过价值分析，结合自上而下确定投资策略和自下而上个券选择的程序，采取久期偏离、收益率曲线配置和类属配置等积极投资策略，发现、确认并利用市场失衡实现组合增值......",
  "info": {
    "url": "无",
    "title": [
      "无"
    ],
    "source_domain": ""
  },
  "score": 0.29443359375
}
```

`FastText/models`可以从 [ChineseWebText](https://github.com/CASIA-LM/ChineseWebText) 项目下载，然后执行 python
FastText/evaluate.py --dates FastText/data,此步骤将为每个数据条目分配 FastText
分数，结果存储在“./FastText/data/fasttext”等目录中。随后，您可以利用这些分数通过使用阈值（默认设置为 0.5）来过滤和提取高质量数据。