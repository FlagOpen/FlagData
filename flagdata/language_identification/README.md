# Data preprocessing phase > Language recognition

LID stands for Language identification, which is a model for language identification.
+ It uses fastText's language classifier, which is trained on Wikipedia, Tatoeba, and SETimes, uses n-grams as features, and uses a hierarchical softmax. 176 languages are classified, and it outputs a score from 0 to 1.
+ Each CPU core can process one thousand documents per second.
+ For each web page, the language classification is done once, and the classification score is obtained. If it is greater than 0.5, then the page is classified as a specific language, otherwise, the page is not sure what language it is in and the page is discarded.

See `split_by_lang.py` for an example.

```markdown
language type: zh, language source:0.99 , original text: 面向大模型研究领域的高效易用数据处理工貝包 .
language type: en, language source:0.57 , original text: Efficient and Easy-to-Use Data Processing Workbags for Large Modeling Research Domain .
language type: ru, language source:1.0 , original text: Эффективные и простые в использовании рабочие пакеты для обработки данных в области исследования больших моделей .
language type: fr, language source:1.0 , original text: Sacs de travail efficaces et faciles à utiliser pour le traitement des données dans le domaine de la recherche sur les grands modèles .
language type: de, language source:0.99 , original text: Effiziente und einfach zu verwendende Datenverarbeitungs-Workbags für große Modellforschungsbereiche .
language type: ja, language source:1.0 , original text: 大規模モデル研究領域のための効率的で使いやすいデータ処理ワークバッグ .
```