# 数据准备阶段>语言识别

LID 代表 Language identification，是语言识别的模型。

+ 使用 fastText 的语言分类器来做分类，fastText 的语言分类器是在 Wikipedia、Tatoeba、SETimes 上面训练的，使用了 n-grams 来作为特征，使用了层级的 softmax。支持 176 种语言的分类，并且最后会输出一个 0~1 的分数。

+ 每个 CPU 核心上，每秒可以处理一千个文档。

+ 对于每一个网页做一次语言分类，得到分类的分数，如果大于 0.5，那么就分类为某个特定的语言，否则表示不确定是什么语言的网页并丢掉这个网页。

具体示例见`split_by_lang.py`

```markdown
language type: zh, language source:0.99 , original text: 面向大模型研究领域的高效易用数据处理工貝包 .
language type: en, language source:0.57 , original text: Efficient and Easy-to-Use Data Processing Workbags for Large Modeling Research Domain .
language type: ru, language source:1.0 , original text: Эффективные и простые в использовании рабочие пакеты для обработки данных в области исследования больших моделей .
language type: fr, language source:1.0 , original text: Sacs de travail efficaces et faciles à utiliser pour le traitement des données dans le domaine de la recherche sur les grands modèles .
language type: de, language source:0.99 , original text: Effiziente und einfach zu verwendende Datenverarbeitungs-Workbags für große Modellforschungsbereiche .
language type: ja, language source:1.0 , original text: 大規模モデル研究領域のための効率的で使いやすいデータ処理ワークバッグ .
```