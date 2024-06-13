# TextCleaner

## Description

FlagData TextCleaner offers a fast and extensible text data cleaning tool. It provides commonly used text cleaning modules. For additional text cleaning features, users can refer to the Operators section.


## Data Format

We support input data in jsonl. As to jsonl format, each line contains a json with key-value pairs.

```
input format:
{
    "id": "bc8a8f4640b153ddaddf154a605fb461",
    "text": "实际软件工程中是否真的需要100%代码覆盖率（code coverage）？\n 实际项目中，项目经理和架构师往往也是不错的测试员，一些严重bug，经常是他们先发现，比测试员还快一点。 项目中有很多的function, 但function之间的重要性是不同的，也就是说，是不均匀的，有的重要，有的没那么重要，同样是80%的覆盖率，一个覆盖到最重要的function，另一个没有，最后的结果也是天差地别的。 和覆盖率相比，更重要的是测试的顺序，确保最常用，最重要，最核心的功能先测试到，有bug，先发现，先解决，这样测试才高效，团队也会越测越有信心。 这也需要测试员对项目和需求有更深入的理解。 覆盖率高，当然好，但工程类的东西往往需要妥协和平衡，时间不够时，先测什么，后测什么，"
}
output format:
{
    "id": "bc8a8f4640b153ddaddf154a605fb461",
    "text": "实际软件工程中是否真的需要100%代码覆盖率（code coverage）？\n 实际项目中，项目经理和架构师往往也是不错的测试员，一些严重bug，经常是他们先发现，比测试员还快一点。 项目中有很多的function, 但function之间的重要性是不同的，也就是说，是不均匀的，有的重要，有的没那么重要，同样是80%的覆盖率，一个覆盖到最重要的function，另一个没有，最后的结果也是天差地别的。 和覆盖率相比，更重要的是测试的顺序，确保最常用，最重要，最核心的功能先测试到，有bug，先发现，先解决，这样测试才高效，团队也会越测越有信心。 这也需要测试员对项目和需求有更深入的理解。 覆盖率高，当然好，但工程类的东西往往需要妥协和平衡，时间不够时，先测什么，后测什么，",
    "clean_text": "实际软件工程中是否真的需要100%代码覆盖率（code coverage）？\n 实际项目中，项目经理和架构师往往也是不错的测试员，一些严重bug，经常是他们先发现，比测试员还快一点。 项目中有很多的function, 但function之间的重要性是不同的，也就是说，是不均匀的，有的重要，有的没那么重要，同样是80%的覆盖率，一个覆盖到最重要的function，另一个没有，最后的结果也是天差地别的。 和覆盖率相比，更重要的是测试的顺序，确保最常用，最重要，最核心的功能先测试到，有bug，先发现，先解决，这样测试才高效，团队也会越测越有信心。 这也需要测试员对项目和需求有更深入的理解。"
}
```

## Default Processors

We provide several useful filters by default. For more data cleaning methods, users can refer to the Operators section to add them.

The default filters will do the following cleaning procedures:

​	1. Remove documents where the proportion of line break characters exceeds 0.25 of the total number of characters.

​	2. Remove sentences containing specific content.

​	3. Remove documents where the ratio of numerical characters is greater than 0.5.

​	4. Replace \xa0 and \u3000 with standard spaces and remove other invisible characters.

​	5. Remove multiple line spaces.