# Operator

We provide some basic operators for data cleaning, filtering, and format conversion to help users build their own data construction processes.

---

## Overview

The provided operators are divided into three types: formatter, pruner, and filter. Formatter is used to process structured data and can realize the conversion between data in different formats; Pruner is used to clean text data; Filter is used for sample filtering.


### Formatter

| Operator  | Description  |
|----------|----------|
| JSONFormatter    | Process json format files   |
| CSVFormatter     | Processcsv format files       |
| ParquetFormatter | Process parquet format files |
| TSVFormatter     | Process tsv format files  |

### Pruner

| Operator  | Description  |
|----------|----------|
| CopyrightPruner      | Clean copyright notices from code samples.                                                                                                                                                                                      |
| EmailPruner          | Clean email addresses from text samples.                                                                                                                                                                                    |
| IpPruner             | Clean IPv4 and IPv6 addresses from text samples. |
| LinkPruner           | Clean HTTP, HTTPS, and FTP links from text samples. |
| NonChineseCharPruner | Remove non-Chinese characters from text samples, with options to keep alphabets, numbers, and punctuation.                                |
| RepeatSentencePruner | Remove repeated sentences in text samples.                                                                                                                                                                 |
| ReplacePruner        | Replace all content in the text that matches a specific regular expression pattern with a designated replacement string.                                   |
| TablePruner          | Remove table texts from text samples. Uses a regular expression to identify and remove text that appears in a table format, within the specified range of column numbers. |
| UnicodePruner        | Normalize both various kinds of whitespaces and unicode punctuations to their standard forms in text samples.                                                                               |

### Filter

| Operator  | Description  |
|----------|----------|
| ActionableVerbNumFilter    | Filter samples based on the presence of actionable verbs within the text, ensuring there are enough such verbs according to a specified minimum count.   |
| Alphanumeric_Ratio_Filter   | Filter samples based on the ratio of alphanumeric characters within the text, ensuring it falls within a specified range.       |
| AvgLineLengthFilter         | Evaluate samples based on the average line length within the text, ensuring it falls within a specified range.    |
| FieldValueFilter            | Filter samples based on the value of a specified field within the data, ensuring it matches the target values.    |
| FlaggedWordsRatioFilter     | Screen samples based on the presence of flagged words, ensuring the ratio of such words in a specified threshold.         |
| LanguageConfidenceFilter    | Filter samples based on language identification confidence scores, ensuring the language matches the specified criteria and the confidence is above a minimum threshold.|
| MaxLineLengthFilter         | Filter samples based on the maximum line length within the text, ensuring it falls within a specified range.        |
| NumericFieldValueFilter     | Filter samples based on numeric values of a specified field within the text, ensuring these values fall within a specified range.              |
| SpecialCharacterRatioFilter | Filter samples based on the ratio of designated characters within the text, ensuring it falls within a specified range.      |
| StopwordRatioFilter         | Filter samples based on the stopword ratio within the text, ensuring the ratio exceeds a specified minimum value.     |
| SuffixFilter                | Filter samples based on the suffix of a specified field within the data, ensuring the field ends with specified suffixes.       |
| TextLengthFilter            | Filter samples based on the length of text within them, ensuring the length is within a specified range.          |
| TokenNumFilter              | Filter samples based on the number of tokens in the text, ensuring the token count is within a specified range.       |
| WordNumFilter               | Filter samples based on the number of words in the text, ensuring the word count is within a specified range.      |
| WordRepetitionRationFilter  | Filter samples based on word-level n-gram repetition ratio, ensuring the repetition does not exceed a specific threshold.     |

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







