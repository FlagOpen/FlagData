### Data enhancement module based on OpenAI interface

The OpenAI interface is utilized to construct a series of single rounds of SFT data for different abilities with three different strategies. The strategies include:

+ ImitateGenerator: augment data using several case samples as templates. Supports simultaneous generation of data in multiple languages.
+ AbilityExtractionGenerator: using the OpenAI interface, generalize the abilities contained in several case samples. Generate new samples and answers based on this collection of capabilities.
+ AbilityDirectGenerator: Generate new samples directly related to a specified ability type or task type. For example, if you specify the ability as "Logical Reasoning", you can generate a series of logical reasoning questions and answers. In order to increase the diversity of generated samples, it is supported to exclude already generated samples.
  
  See `example.py` for an example.

 ```python
example_ls = [
    '我小时候吃虾，然后出现过敏反应。几年后，我吃了带有章鱼的虾，然后出现了更严重的过敏反应。每次我去那些有强烈虾味的餐馆，我都会一贯性地出现过敏反应。这可以推断出：',
    '小明观察到他周围的人们都喜欢吃巧克力。因此，他得出结论：所有人都喜欢吃巧克力。请问这个结论是否可靠？如果可靠，可以如何解释这种观察结果？如果不可靠，应该如何修正这个结论？',
    '在一个实验中，A组实验者使用药物X，B组实验者使用药物Y。最后发现A组的痊愈率明显高于B组。基于这一结果，我们可以得出什么结论？',
    '如果一个人经常迟到，那么他的时间观念是强还是弱？',
    '如果一个地区的犯罪率降低了，那么这个地区的治安是提高了还是降低了？']

task = '逻辑推理'
 ```