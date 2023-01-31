# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from flagdata.analysis.text_analyzer import CoreNLPAnalyzer

with CoreNLPAnalyzer(url="https://corenlp.run", lang="zh", annotators="parse") as analyzer:
    text = "随着大规模预训练模型在人工智能领域所取得的成功，相关研究对数据规模和数据处理工具提出了更高的要求。因此我们推出了FlagData，一个便于使用且易于扩展的数据处理工具包。"
    ana_result = analyzer.analyze(text)
    print("*" * 10, "tokens", "*" * 10)
    print(ana_result.tokens)  # tokens
    print("*" * 10, "parse", "*" * 10)
    print(ana_result.parse_tree)  # parse
    print("*" * 10, "binaryParse", "*" * 10)
    print(ana_result.bi_parse_tree)  # binaryParse
    print("*" * 10, "basicDependencies", "*" * 10)
    print(ana_result.basic_dep)  # basicDependencies
    print("*" * 10, "enhancedDependencies", "*" * 10)
    print(ana_result.enhanced_dep)  # enhancedDependencies
    print("*" * 10, "enhancedPlusPlusDependencies", "*" * 10)
    print(ana_result.enhanced_pp_dep)  # enhancedPlusPlusDependencies
    print("*" * 10, "entitymentions", "*" * 10)
    print(ana_result.entities)  # entitymentions
    print("*" * 10, "openie", "*" * 10)
    print(ana_result.openie)  # openie
    print("*" * 10, "prettyPrintParseTree", "*" * 10)
    print(analyzer.pretty_print_tree(ana_result.parse_tree[0]))  # pretty print
    # print(ana_result.ann_result) # original server's response format
