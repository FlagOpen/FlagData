# TextAnalyzer

## Description
We provide an up-to-date, user-friendly NLP analysis tool based on Stanford [CoreNLP](https://stanfordnlp.github.io/CoreNLP/), an nlp tool for natural language processing in Java. 
CoreNLP provides a lingustic annotaion pipeline, which means users can use it to tokenize, ssplit(sentence split), POS, NER, constituency parse, dependency parse, openie etc. However, it's written in Java, which can not be interacted directly with Python programs. Therefore, we developed an analyzer client in Python.
The FlagData Analyzer can be used to start a CoreNLP Server once you've followed the official [release](https://stanfordnlp.github.io/CoreNLP/download.html) and download necessary packages and corresponding models. Or, if a server is already started, the only thing you need to do is to specify the server's url (such as CoreNLP's official server url https://corenlp.run), and call the `analyze()` method. 

## Installation
[Java 8](https://www.oracle.com/java/technologies/javase-downloads.html) is required if you want to start the CoreNLP server locally.

Python >= 3.9

Install the Analysis modules via pip. By doing this, only the specified module's dependencies will be installed. This is for users who only want to use the Analysis module and do not want to install the other dependencies:
```bash
pip install flagdata[analysis]
```
If you want to install all the dependencies, use the following command:
```bash
pip install flagdata[all]
```



## Usage

### Quick Start

Sometimes you may want to get tokenized (pos, ner) result directly without manual extraction from original output. Thus, we provide `tokenize(), pos_tag(), ner()` methods to simplify the whole process.

```python
from flagdata.analysis.text_analyzer import CoreNLPAnalyzer
analyzer = CoreNLPAnalyzer(url="https://corenlp.run", annotators="ner", lang="en")
data = "FlagData is a fast and extensible toolkit for data processing provided by BAAI. Enjoy yourself! "
tokenized_text = analyzer.tokenize(data)
pos_tags = analyzer.pos_tag(data)
ners = analyzer.ner(data)
analyzer.close()
print(tokenized_text)
# [['FlagData', 'is', 'a', 'fast', 'and', 'extensible', 'toolkit', 'for', 'data', 'processing', 'provided', 'by', 'BAAI', '.'], ['Enjoy', 'yourself', '!']]
print(pos_tags)
# [['NNP', 'VBZ', 'DT', 'JJ', 'CC', 'JJ', 'NN', 'IN', 'NN', 'NN', 'VBN', 'IN', 'NN', '.'], ['VB', 'PRP', '.']]
print(ners)
# [[{('BAAI', (74, 78)): 'ORGANIZATION'}], []]
```

**Start a New Server and Analyze Text**

If you want to start a server locally, it's more graceful to use `with ... as ...` to handle exceptions.

```python
from flagdata.analysis.text_analyzer import CoreNLPAnalyzer
# max_mem: max memory use, default is 4. threads: num of threads to use, defualt is num of cpu cores.
with CoreNLPAnalyzer(annotators="tokenize", corenlp_dir="/path/to/corenlp", local_port=9000, max_mem=4, threads=2) as analyzer:
    # your code here
```

**Use an Existing Server**

You can also use an existing server by providing the url.

```python
from flagdata.analysis.text_analyzer import CoreNLPAnalyzer
# lang for language, default is en.
# you can specify annotators to use by passing `annotator="tokenize,ssplit"` args to CoreNLP. If not provided, all available annotators will be used.
with CoreNLPAnalyzer(url="https://corenlp.run", lang="en") as analyzer:
    # your code here
```

### Advanced Usage

For advanced users, you may want to have access to server's original response in dict format:

```python
ana_result = analyer.analyze("CoreNLP is your one stop shop for natural language processing in Java! Enjoy yourself! ")
print("*" * 10, "tokens", "*" * 10)
print(ana_result.tokens) # tokens
print("*" * 10, "parse", "*" * 10)
print(ana_result.parse_tree) # parse
print("*" * 10, "binaryParse", "*" * 10)
print(ana_result.bi_parse_tree) # binaryParse
print("*" * 10, "basicDependencies", "*" * 10)
print(ana_result.basic_dep) # basicDependencies
print("*" * 10, "enhancedDependencies", "*" * 10)
print(ana_result.enhanced_dep) # enhancedDependencies
print("*" * 10, "enhancedPlusPlusDependencies", "*" * 10)
print(ana_result.enhanced_pp_dep) # enhancedPlusPlusDependencies
print("*" * 10, "entitymentions", "*" * 10)
print(ana_result.entities) # entitymentions
print("*" * 10, "openie", "*" * 10)
print(ana_result.openie) # openie
print("*" * 10, "prettyPrintParseTree", "*" * 10)
print(analyzer.pretty_print_tree(ana_result.parse_tree[0])) # pretty print 
print(ana_result.ann_result) # original server's response format
```

## Extra Notes

- If you choose to start server locally, it'll take a while to load models for the first time you analyze a sentence.

- For timeout error, a simple retry may be useful. Otherwise, you should make sure the annotators and language you specified are supported by CoreNLP server.

- Also, if "with"(context manager) is not used, remember to call close() method to stop the Java CoreNLP server. 

## Reference
Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. The Stanford CoreNLP Natural Language Processing Toolkit In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 55-60. \[[pdf](https://nlp.stanford.edu/pubs/StanfordCoreNlp2014.pdf)\]


