选择BERT和fasttext作为评估模型，是因为它们具有以下优点：

1. BERT模型在文本分类和理解任务中表现出色，具有强大的语言理解和表示能力，能够有效地评估文本质量。
2. FastText模型具有高效的训练和推理速度，同时保持分类性能，可以显著减少训练和推理时间。

文章比较了不同的文本分类模型，包括逻辑回归、BERT和FastText，以评估它们的性能。在实验中，BERTEval和FastText模型在文本分类任务中表现良好，其中FastText模型在精度和召回率方面表现最佳。【实验结果来自ChineseWebText】


![Classification results of different evaluation models.](quality_assessment.png)