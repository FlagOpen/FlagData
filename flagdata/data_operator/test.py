import sys
import os

# 动态添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flagdata.data_operator.formatter import *
from flagdata.data_operator.samplefilter import *
from flagdata.data_operator.pruner import *

csv_formatter = CSVFormatter()
json_formatter = JSONFormatter()
tsv_formatter = TSVFormatter()
parquet_formatter = ParquetFormatter()

json_formatter.format_data('test_data.json', 'test_data.csv', csv_formatter)
json_formatter.format_data('test_data.json', 'test_data.tsv', tsv_formatter)
json_formatter.format_data('test_data.json', 'test_data.parquet', parquet_formatter)

# to json
csv_formatter.format_data('test_data.csv', 'new_data.json', json_formatter)
tsv_formatter.format_data('test_data.tsv', 'new_data.json', json_formatter)
parquet_formatter.format_data('test_data.parquet', 'new_data.json', json_formatter)

# to csv
json_formatter.format_data('test_data.json', 'new_data.csv', csv_formatter)
tsv_formatter.format_data('test_data.tsv', 'new_data.csv', csv_formatter)
parquet_formatter.format_data('test_data.parquet', 'new_data.csv', csv_formatter)

# to tsv
json_formatter.format_data('test_data.json', 'new_data.tsv', tsv_formatter)
csv_formatter.format_data('test_data.csv', 'new_data.tsv', tsv_formatter)
parquet_formatter.format_data('test_data.parquet', 'new_data.tsv', tsv_formatter)

# to parquet
json_formatter.format_data('test_data.json', 'new_data.parquet', parquet_formatter)
csv_formatter.format_data('test_data.csv', 'new_data.parquet', parquet_formatter)
tsv_formatter.format_data('test_data.tsv', 'new_data.parquet', parquet_formatter)

pruner = EmailPruner(text_key='content')
samples = [
    {'content': "Please contact us at support@example.com for more information."},
    {'content': "Send an email to admin@example.org or info@example.com."},
    {'content': "This text contains no email addresses."},
    {'content': "Emails such as test.user+label@example.co.uk and user.name@domain.com should be removed."}
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
print(cleaned_samples)

pruner = LinkPruner(text_key='content')
samples = [
    {'content': "Check out this website: https://example.com for more information."},
    {'content': "You can also visit our FTP server at ftp://ftp.example.org."},
    {'content': "This text contains no links at all."},
    {'content': "Multiple links: http://example.com and https://example.org should be removed."}
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
print(cleaned_samples)

pruner = NonChineseCharPruner(text_key='content', keep_alphabet=False, keep_number=True, keep_punc=True)
samples = [
    {'content': "This is a test. 这是一个测试。123!"},
    {'content': "Removing non-Chinese characters: 去除非中文字符。"},
    {'content': "保持字母和数字: Keep alphabets and numbers 123."},
    {'content': "混合文本: Mixed text with 中文 and English."}
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
print(cleaned_samples)

pruner = RepeatSentencePruner(text_key='content', lowercase=True, ignore_special_character=True,
                              min_repeat_sentence_length=2)
samples = [
    {'content': "This is a test. This is a test. This is only a test."},
    {'content': "重复的句子。重复的句子。"},
    {'content': "No repetition here. Everything is unique."},
    {'content': "Special characters! Special characters! Should be ignored."}
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
print(cleaned_samples)

pruner = ReplacePruner(
    text_key='content',
    pattern=[r'\bfoo\b', r'\bbar\b'],
    repl=['baz', 'qux']
)
samples = [
    {'content': "This is a foo test. The bar should be replaced."},
    {'content': "Foo and Bar are common placeholders."},
    {'content': "No matches here."},
    {'content': "Another example with foo and bar."}
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
print(cleaned_samples)

pruner = TablePruner(text_key='content', min_col=3, max_col=5)
samples = [
    {'content': "This is a normal text.\nHeader1\tHeader2\tHeader3\nData1\tData2\tData3\nMore text here."},
    {'content': "Normal text without table."},
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
print(cleaned_samples)

pruner = UnicodePruner(text_key='content')
samples = [
    {'content': "这是一个测试，包含多种标点符号和空白字符。"},
    {'content': "Normalize spaces and unicode punctuations… Here is an example： '这是一个例子'。"},
    {'content': "特殊字符： ！@#【】％，普通字符： abc123..."},
    {'content': "标准化空格\t和各种标点符号。—这是另一个例子。"}
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
print(cleaned_samples)

pruner = CopyrightPruner(text_key='content')
samples = [
    {'content': "/*\nThis is a multi-line comment\nwith COPYRIGHT notice.\n*/\nint main() { return 0; }"},
    {'content': "// Copyright 2022 by Author\nint main() { return 0; }"},
    {'content': "# COPYRIGHT 2022\nint main() { return 0; }"},
    {'content': "int main() { return 0; } // No copyright here"}
]
cleaned_samples = [pruner.process(sample.copy()) for sample in samples]
print(cleaned_samples)

filter = MaxLineLengthFilter(text_key='content', min_length=20, max_length=100)
samples = [
    {'content': "This line is short.\nThis is a much longer line of text that should be within the limits."},
    {'content': "Short line.\nAnother short line.\nYet another one."},
    {'content': "A reasonably sized line.\nThis one is toooooooooooooooooooooooooooooooooooooo long to be kept."}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = LanguageConfidenceFilter(text_key='content', languages=['en', 'fr'], min_confidence=0.85)
samples = [
    {'content': "This is an English text."},
    {'content': "Ceci est un texte français."},
    {'content': "Dies ist ein deutscher Text."}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = FlaggedWordsRatioFilter(text_key='content', min_ratio=0.00, max_ratio=0.05, flagged_words_list=['badword'])
samples = [
    {'content': "This is a clean example."},
    {'content': "This example contains some badword and another badword."},
    {'content': "Another clean example, well maybe not so badword."}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = AvgLineLengthFilter(text_key='content', min_length=20, max_length=100)
samples = [
    {'content': "This is a line.\nThis is a longer line of text that exceeds the average length."},
    {'content': "Short line.\nAnother short line."},
    {'content': "This line is okay.\nThis line is perfectly fine and within range.\nThis line is also good."}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = AlphanumericRatioFilter(text_key='content', min_ratio=0.2, max_ratio=0.8)
samples = [
    {'content': "Hello, world! This is a test sample. 12345"},
    {'content': "!!!$$$%%% ^^^&&& ***((()))"},
    {'content': "Data2020 with some more numbers 3456 and text"},
    {'content': "Hello, world! This is a test sample."},
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = WordRepetitionRationFilter(text_key='content', ngram_length=3, min_ratio=0.0, max_ratio=0.2)
samples = [
    {'content': "The cat chased the mouse around the house."},
    {'content': "Happy birthday, happy birthday, happy birthday to you!"},
    {'content': "This sentence is unique without any repetition within the sentence itself."},
    {'content': "祝你生日快乐，祝你生日快乐。"}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = WordNumFilter(text_key='content', min_words=10, max_words=50)
samples = [
    {'content': "This is too short."},
    {'content': "This example should have just enough words to pass the minimum threshold but not exceed the maximum."},
    {
        'content': "This text contains an excessively large number of words formed from multiple sentences, and it continues adding more content than necessary for a standard sample, therefore likely surpassing the upper limit."}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = TextLengthFilter(text_key='content', min_length=20, max_length=200)
samples = [
    {'content': "Short text."},
    {
        'content': "This is a much longer piece of text that should meet the minimum length requirement but not exceed the maximum."},
    {
        'content': "This text is way too long. " * 10 + "It repeats a lot, making it significantly longer than what would be acceptable within the set bounds of this particular example. The length of this text should clearly surpass the upper limit set by the pruner."}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = ActionableVerbNumFilter(text_key='content', min_verbs=2)
samples = [
    {'content': "He runs and she jumps."},
    {'content': "Consider implementing changes."},
    {'content': "Beautiful scenery with no actions here."},
    {'content': "猫擅长捉老鼠"},
    {'content': "篮球运动员需要在比赛前进行热身，以保证在高强度的对抗下不会受伤。"},
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = SuffixFilter(field_key='filename', allowed_suffixes=['.txt', '.md', '.pdf'])
samples = [
    {'filename': "example.txt"},
    {'filename': "report.pdf"},
    {'filename': "image.jpeg"},
    {'filename': "notes.md"}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = StopwordRatioFilter(text_key='content', language='en', min_ratio=0.1)
samples = [
    {'content': "This is an example of a text with very few stopwords."},
    {'content': "This, however, is a text where the and it are used extensively."},
    {'content': "An example without much content relevance."}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = NumericFieldValueFilter(key_path='data.numeric_field', min_value=10, max_value=100)
samples = [
    {'data': {'numeric_field': '25'}},
    {'data': {'numeric_field': '105'}},
    {'data': {'numeric_field': 'not_a_number'}},
    {'data': {'numeric_field': '50'}}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = FieldValueFilter(key_path='metadata.status', valid_values=['approved', 'validated'])
samples = [
    {'content': "Sample data", 'metadata': {'status': 'approved'}},
    {'content': "More sample data", 'metadata': {'status': 'pending'}},
    {'content': "Additional data", 'metadata': {'status': 'validated'}}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = SpecialCharacterRatioFilter(text_key='content', min_ratio=0.00, max_ratio=0.1, characters='@^&*',
                                     add_default=True)
samples = [
    {'content': "Hello world! This is an example."},
    {'content': "Welcome!!! #Amazing @Day."},
    {'content': "Check this out: ^&*()@^&* Special chars everywhere!!!"}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)

filter = TokenNumFilter(text_key='content', tokenizer_name='bert-base-uncased', min_tokens=10, max_tokens=50)
samples = [
    {'content': "Short text."},
    {'content': "This is a slightly longer piece of text that should exceed the minimum token count."},
    {
        'content': "This piece of text is exceptionally long and contains many tokens, potentially exceeding the maximum token count allowed in this example. This piece of text is exceptionally long and contains many tokens, potentially exceeding the maximum token count allowed in this example. This piece of text is exceptionally long and contains many tokens, potentially exceeding the maximum token count allowed in this example. This piece of text is exceptionally long and contains many tokens, potentially exceeding the maximum token count allowed in this example."}
]
samples = [sample for sample in samples if filter.process(sample)]
print(samples)
