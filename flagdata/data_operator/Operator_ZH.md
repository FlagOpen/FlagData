# 算子

我们提供了一些用于数据清洗，过滤，格式转换等的基本算子，帮助用户构建自己的数据构建流程。

---

## Overview

提供的算子分为三种：Formatter、Pruner、Filter。Formatter用于处理结构化数据，可以用于不同格式数据的相互转换；Pruner用于清洗文本数据；Filter用于样本过滤。

### Formatter

| 算子          | 描述                       |
|-----------------|--------------------------|
| JSONFormatter  | 处理 JSON 格式文件        |
| CSVFormatter  | 处理 CSV 格式文件         |
| ParquetFormatter | 处理 Parquet 格式文件    |
| TSVFormatter  | 处理 TSV 格式文件         |

### Pruner

| 操作符               | 描述                                                                                                                 |
|----------------------|--------------------------------------------------------------------------------------------------------------------|
| CopyrightPruner      | 清除代码样本中的版权声明。                                                                                          |
| EmailPruner          | 清除文本样本中的电子邮件地址。                                                                                      |
| IpPruner             | 清除文本样本中的 IPv4 和 IPv6 地址。                                                                                |
| LinkPruner           | 清除文本样本中的 HTTP、HTTPS 和 FTP 链接。                                                                         |
| NonChineseCharPruner | 移除文本样本中的非中文字符，可选择保留字母、数字和标点符号。                                                        |
| RepeatSentencePruner | 移除文本样本中的重复句子。                                                                                          |
| ReplacePruner        | 将文本中符合特定正则表达式模式的所有内容替换为指定的替换字符串。                                                   |
| TablePruner          | 移除文本样本中的表格文本。使用正则表达式识别并移除表格格式文本，指定列数范围内。                                      |
| UnicodePruner        | 规范化文本样本中的各种空白字符和 Unicode 标点符号，使其成为标准形式。                                                |


### Filter

| 操作符                        | 描述                                                                                                  |
|-------------------------------|-----------------------------------------------------------------------------------------------------|
| ActionableVerbNumFilter       | 基于文本中可执行动词的数量过滤样本，确保符合指定的最小数量要求。                                       |
| Alphanumeric_Ratio_Filter     | 基于文本中字母数字字符比例过滤样本，确保其比例在指定范围内。                                           |
| AvgLineLengthFilter           | 基于文本中平均行长度评估样本，确保其长度在指定范围内。                                                 |
| FieldValueFilter              | 基于数据中指定字段的值过滤样本，确保其值与目标值匹配。                                                 |
| FlaggedWordsRatioFilter       | 基于文本中标记词的比例筛选样本，确保其比例在指定阈值内。                                               |
| LanguageConfidenceFilter      | 基于语言识别置信度分数过滤样本，确保语言符合指定标准且置信度高于最低阈值。                            |
| MaxLineLengthFilter           | 基于文本中最大行长度过滤样本，确保其长度在指定范围内。                                                 |
| NumericFieldValueFilter       | 基于文本中指定字段的数值过滤样本，确保其数值在指定范围内。                                             |
| SpecialCharacterRatioFilter   | 基于文本中指定字符的比例过滤样本，确保其比例在指定范围内。                                             |
| StopwordRatioFilter           | 基于文本中停用词比例过滤样本，确保其比例超过指定的最小值。                                             |
| SuffixFilter                  | 基于数据中指定字段的后缀过滤样本，确保其字段以指定后缀结尾。                                           |
| TextLengthFilter              | 基于文本长度过滤样本，确保其长度在指定范围内。                                                         |
| TokenNumFilter                | 基于文本中的标记数量过滤样本，确保其标记数在指定范围内。                                               |
| WordNumFilter                 | 基于文本中的单词数量过滤样本，确保其单词数在指定范围内。                                               |
| WordRepetitionRationFilter    | 基于单词级n-gram重复比例过滤样本，确保其重复比例不超过指定阈值。                                       |


## Examples

### Formatter

`````
csv_formatterr = CSVFormatter()   
json_formatter = JSONFormatter() 
tsv_formatter = TSVFormatter() 
parquet_formatter = ParquetFormatter()

#csv to json
csv_formatter.format_data('test_data.csv', 'new_data.json', json_formatter)
#parquet to csv
parquet_formatter.format_data('test_data.parquet', 'new_data.csv', csv_formatter)
`````

### Pruner

`````
pruner = EmailPruner(text_key="content")
samples = [
    {"content": "Please contact us at support@example.com for more information."},
    {"content": "Send an email to admin@example.org or info@example.com."},
    {"content": "This text contains no email addresses."},
    {"content": "Emails such as test.user+label@example.co.uk and user.name@domain.com should be removed."}
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
'''
Expected Output:
[{"content": "Please contact us at  for more information."}, 
{"content": "Send an email to  or ."},
{"content": "This text contains no email addresses."},
{"content": "Emails such as  and  should be removed."}]
'''
`````

### Filter

`````
sample_filter = Alphanumeric_Ratio_Filter(text_key="content", min_ratio=0.2, max_ratio=0.8)
samples = [
    {"content": "Hello, world! This is a test sample. 12345"},  
    {"content": "!!!$$$%%% ^^^&&& ***((()))"},                  
    {"content": "Data2020 with some more numbers 3456 and text"},
    {"content": "Hello, world! This is a test sample."},  
]
samples = [sample for sample in samples if filter.process(sample)]
'''
Expected Output: [{"content": "Hello, world! This is a test sample. 12345"},
{"content": "Hello, world! This is a test sample."}]
'''
`````







