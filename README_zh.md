<div id=top align="center">

![FlagData](flagdata_logo.png)
[![Pypi Package](https://img.shields.io/pypi/v/flagdata?label=pypi%20package)](https://pypi.org/project/flagdata/)
[![Python Application](https://github.com/FlagOpen/FlagData/actions/workflows/python-app.yml/badge.svg)](https://github.com/FlagOpen/FlagData/actions/workflows/python-app.yml)
[![License](https://img.shields.io/github/license/FlagOpen/FlagData.svg?color=blue)](https://github.com/FlagOpen/FlagData/blob/main/LICENSE)
![GitHub release (release name instead of tag name)](https://img.shields.io/github/v/release/FlagOpen/FlagData?include_prereleases&style=social)

   | [English](README.md) | [中文](README_zh.md) |

</div>

-----------------------------------------------------------------------
数据是人工智能领域发展的基础要素之一。随着大规模预训练模型及相关技术不断取得突破，在相应研究中使用高效数据处理工具提升数据质量变得越来越重要。因此我们推出了FlagData，一个便于使用且易于扩展的数据处理工具包。FlagData集成了包含清洗、标注、压缩、统计分析等功能在内的多个数据处理工具与算法，为自然语言处理、计算机视觉等领域的模型训练与部署提供了数据层面的有力支撑。

FlagData支持以下特性：

* 安装后简单配置即可上手使用，低代码量实现自定义功能。

* 基于数据蒸馏算法压缩模型训练数据，达到与全量数据训练可比的效果。

* 可从原始html/text快速清洗得到高质量结构化数据，注重敏感信息滤除，避免隐私泄露风险。

* 支持自然语言处理和计算机视觉领域多项任务的数据标注，标注结果集方便读取。


## 动态
- [3rd Jan 2023] FlagData v1.0.0 上线了!

--------------------------------------------------------------------------------

- [安装](#安装)
- [快速上手](#快速上手)
    - [数据清洗](#数据清洗)
    - [数据分析](#数据分析)
    - [数据蒸馏](#数据压缩)
    - [数据标注](#数据标注)
- [配置](#配置)
- [使用指南](#使用指南)
- [联系我们](#联系我们)
- [参考项目](#参考项目)
- [许可证](#许可证)

## 安装
- Python 版本 >= 3.9
- Pytorch 版本 >= 1.11 如果你需要使用数据压缩模块，请参考[Pytorch官方网站](https://pytorch.org/get-started/locally/)安装与你的运行环境适配的Pytorch版本。
- 可选: 如需使用数据标注工具，请先安装[NGINX](https://www.nginx.com/resources/wiki/start/topics/tutorials/install/)，然后根据[快速上手](#快速上手)中的步骤配置应用。
通过pip安装FlagData所有模块。通过下面的命令，你将安装所有模块的依赖包。
```bash
pip install flagdata[all]
```

选择性安装FlagData中所需的模块 (`module` 需要替换成模块名，如 `cleaner`、`condensation`、`analysis`等)。你将只会安装对应模块的依赖包，这适合那些只想使用某个特定模块且不想安装其他模块依赖包的使用者。
```bash
pip install flagdata[module]
```

**安装main分支的最新版本**

如果你想安装/更新到main分支的最新版本，请使用以下命令：
```
git clone https://github.com/cofe-ai/FlagData.git
pip install .[all]
```

**基于源码二次开发**
```bash
git clone https://github.com/cofe-ai/FlagData.git
pip install -r requirements.txt
```

## 快速上手

### 数据清洗
使用FlagData的数据清洗功能仅需两步：

1. 修改YAML配置文件中的数据路径与格式。我们在配置文件模板中为每个参数给出了详细的注释来解释其含义。同时你也可以参考[配置](#配置)章节。

2. 在以下代码中指定配置文件路径，运行即可
   ```python
   from flagdata.cleaner.text_cleaner import DataCleaner
   if __name__ == "__main__": # 多进程中主模块安全导入
      cleaner = DataCleaner("config.yaml")
      cleaner.clean()
   ```

清洗后的文件会以`jsonl`的格式保存到配置文件中指定的`output`参数对应的路径。

### 数据分析
使用FlagData的数据分析功能最便捷的方式是利用我们提供的客户端请求CoreNLP官方的服务，示例代码如下：

```python
from flagdata.analysis.text_analyzer import CoreNLPAnalyzer
# 创建客户端以调用官方的demo服务
analyzer = CoreNLPAnalyzer(url="https://corenlp.run", lang="en")
data = "FlagData is a fast and extensible toolkit for data processing provided by BAAI. Enjoy yourself! "
tokenized_text = analyzer.tokenize(data)
print(tokenized_text)
# [['FlagData', 'is', 'a', 'fast', 'and', 'extensible', 'toolkit', 'for', 'data', 'processing', 'provided', 'by', 'BAAI', '.'], ['Enjoy', 'yourself', '!']]
pos_tags = analyzer.pos_tag(data)
print(pos_tags)
# [['NNP', 'VBZ', 'DT', 'JJ', 'CC', 'JJ', 'NN', 'IN', 'NN', 'NN', 'VBN', 'IN', 'NN', '.'], ['VB', 'PRP', '.']]
ners = analyzer.ner(data)
print(ners)
# [[{('BAAI', (74, 78)): 'ORGANIZATION'}], []]
analyzer.close()
```

### 数据压缩
使用FlagData的数据压缩功能仅需两步：

1. 修改YAML配置文件的路径等参数。我们在配置文件模板中为每个参数给出了详细的注释来解释其含义。同时你也可以参考[Configuration](#configuration)章节。

2. 在以下代码中指定配置文件路径，运行即可
   
   ```python
   from flagdata.condensation.data_distillation import DataDistillationTrainer
   # 需要在此处指定修改后的配置文件路径
   trainer = DataDistillationTrainer("config/distillation_config.yaml") 
   # 默认数据格式为jsonl，包含"text"和"label"两个键值对
   trainer.load_data()
   # fit() 方法将会运行数据压缩算法并以二进制的格式保存压缩后的数据，可以使用torch.load()读取
   # 可以通过设置配置文件中的"distilled_save_path"自定义保存路径
   trainer.fit()
   ```

### 数据标注

1. 将`flagdata/annotation/dist`放到nginx默认的`html`下

2. 修改`nginx.confg`以添加`location`
   ```
   location / {
       root /{your html path}/dist;   # change
       index index.html index.htm;
       try_files $uri $uri/ /index.html;
   }
   ```

3. 重新启动nginx

4. 根据nginx配置的ip地址访问标注系统

## 配置
针对`数据清洗`、`数据压缩`模块， 我们提供了配置文件模板：[cleaner_config.yaml](https://dorc.baai.ac.cn/resources/projects/FlagData/cleaner_config.yaml)， [distillation_config.yaml](https://dorc.baai.ac.cn/resources/projects/FlagData/distillation_config.yaml)。 配置文件为易读的 [YAML](https://yaml.org) 格式，并提供了详尽的注释。使用这些模块前请确认已经在配置文件中修改好相应参数。

以下是一些你需要注意的重要参数：

### 数据清洗

   ```yaml
   # 待清洗的原始数据
   input: ./demo/demo_input.jsonl
   # 清洗后数据的保存路径
   output: ./demo/output.jsonl
   ```

### 数据压缩

   ```yaml
   train_data_path: <训练数据路径>
   test_data_path: <测试数据路径>
   # huggingface上的预训练模型，目录下需要包含pytorch_model.pt, vocab.txt等
   model_name: "/data/scripts/pretrained_models/distilbert-base-uncased"
   # model.fit()方法将会运行数据压缩算法并以二进制的格式保存压缩后的数据，可以使用torch.load()读取
   distilled_save_path: <path to save distilled data>
   # 可选: 加载压缩后的数据，用于初始化或继续训练
   distilled_load_path: null
   ```

## 使用指南

我们提供了一系列使用指南，帮助用户快速体验FlagData的特性。
* [Tutorial 1: 清洗从互联网上获取到的原始文本](/docs/tutorial_01_cleaner.md)
* [Tutorial 2: 分析、处理文本数据](/docs/tutorial_02_analysis.md)
* [Tutorial 3: 使用数据压缩算法减少数据使用量](/docs/tutorial_03_condensation.md)
* [Tutorial 4: NLP任务标注](/docs/tutorial_04_text_annotation.md)
* [Tutorial 5: CV任务标注](/docs/tutorial_05_image_annotation.md)

## 联系我们
如果你对本项目的使用和代码有任何问题，可以提交issue。同时你也可以通过邮箱 data@baai.ac.cn 直接联系我们

## 参考项目
本项目部分参考自以下代码：
[GeneralNewsExtractor](https://github.com/GeneralNewsExtractor/GeneralNewsExtractor), 
[text-data-distillation](https://github.com/arumaekawa/text-dataset-distillation), 
[emoji](https://github.com/carpedm20/emoji),
[transformers](https://github.com/huggingface/transformers)。

## 许可证
FlagData项目整体基于 [Apache 2.0 协议](LICENSE)。此外，部分代码需要遵循下面的协议：
- GeneralNewsExtractor 基于[GPL v3.0 license](https://github.com/GeneralNewsExtractor/GeneralNewsExtractor/blob/master/LICENSE)
- emoji 基于[BSD license](https://github.com/carpedm20/emoji/blob/master/LICENSE.txt)
- transformers 基于[Apache 2.0 license](https://github.com/huggingface/transformers/blob/main/LICENSE)
