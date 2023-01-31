# TextCleaner

## Description
FlagData Cleaner provides a fast, extensible data cleaning tool for web text. It provides most of the commonly used text cleaning modules which can be easily configured by a human-readable YAML config file. Meanwhile, users can define their own processors and add to the whole pipeline.

## Installation
Python >= 3.9

Install the Cleaner modules via pip. By doing this, only the specified module's dependencies will be installed. This is for users who only want to use the Cleaner module and do not want to install the other dependencies:
```bash
pip install flagdata[cleaner]
```
If you want to install all the dependencies, use the following command:
```bash
pip install flagdata[all]
```

## Usage
We provide two basic modules, `Extractor` and `Filter`. Extractors are used when you need to extract information (title, publish time and content) from the text of html format. Filters are used to clean up the content extracted from html. You can enable both of them as a pipeline or use one of them for extraction/cleaning. There're serveral processors under extractors/filters, which can be disabled by commenting or removal. You can also add your own extractors/filters by inheriting the basic class. Note that the order of processors in the config file matters as they will be executed sequentially.

### Data Format
We support input data in jsonl/plain text (if input data is plain text, set `is_jsonl` to `false` in config file). However, if you want to extract text from html, then jsonl is required.
As to jsonl format, each line contains a json with key-value pairs. you can specify which one need to be cleaned by setting `source_key` in config file. You may orgnize your data like this: 
```json
{"rawContent": "<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta property="article:published_time" content="2022-12-05T00:52:04+08:00">\n    <title>An Introduction to BAAI</title>\n</head>\n<body>\n<p>With the support of Ministry of Science and Technology..."}
{"rawContent": "<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta property="article:published_time" content="2022-12-05T00:52:04+08:00">\n    <title>BAAI CONFERENCE</title>\n</head>\n<body>\n<p>The BAAI Conference is an annual international high-end academic exchange event in ..."}
```

The output data will always be in jsonl format with cleaned content corresponding to `result_key` in config file.

### Default Processors
We provide several useful extractors and filters by default, and you can disable them by commenting. 

The default extractors will locate and extract title, content and publish time from html and delete useless information such as text in navigation bars. The default filters will do the following cleaning procedures:
1. convert traditional Chinese to simplified version
2. filter out emojis and control characters
3. remove private information and url links
4. remove redundant spaces/line breaks
5. check text integrity and clip it by end marks

### Quick start
There are basically 2 steps in order to use our FlagData Cleaner:
1. Modify the YAML config file according to your data format. We have written detailed comments in the configuration file to explain the meaning of each parameter.

2. Specify the path to the config file and run!
    ```python
    from flagdata.cleaner.text_cleaner import DataCleaner
    # safe importing of main module in multi-processing
    if __name__ == "__main__": 
        # you need to specify your own configuration file path
        cleaner = DataCleaner("config.yaml")
        cleaner.clean()
    ```
The cleaned data will be saved to the corresponding path in `jsonl` format according to the `output` parameter in the configuration file.

### Advanced Usage

If you want to define your own processor, following these steps:
1. Inherit the basic processor class(BasicExtractor/BasicCleaner) to define yours.
    ```python
    from flagdata.cleaner.utils.filter import BasicCleaner
    class MyFakeFilter(BasicCleaner):
        def __init__(self, **args):
            pass # your code here

        def clean_article(self, article):
            cleaned_article = article # your code here
            return cleaned_article
    ```
2. Add your custom class and related parameters to the configuration file.

3. Import your class (or you can define your own class and run in a single file) and run the code:
    ```python
    from flagdata.cleaner.text_cleaner import DataCleaner
    from xxx import MyFakeFilter
    # specify your own configuration file path
    cleaner = DataCleaner("config.yaml")
    cleaner.clean()
    ```

## Config
We use [YAML](https://yaml.org/) format configuration file for readability. For more information about YAML, you can check https://yaml.org/ . 

Descriptions and details of each parameters are commented in our [config template](https://dorc.baai.ac.cn/resources/projects/FlagData/cleaner_config.yaml). 
User-defined extractors/filters can be added following the same format. Always remember to check whether the class name and init params are the same as the ones in the config file. 

## Reference
1. https://github.com/GeneralNewsExtractor/GeneralNewsExtractor
2. https://github.com/carpedm20/emoji
