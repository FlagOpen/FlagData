### Data enhancement module based on OpenAI interface

The OpenAI interface is utilized to construct a series of single rounds of SFT data for different abilities with three different strategies. The strategies include:

+ ImitateGenerator: augment data using several case samples as templates. Supports simultaneous generation of data in multiple languages.
+ AbilityExtractionGenerator: using the OpenAI interface, generalize the abilities contained in several case samples. Generate new samples and answers based on this collection of capabilities.
+ AbilityDirectGenerator: Generate new samples directly related to a specified ability type or task type. For example, if you specify the ability as "Logical Reasoning", you can generate a series of logical reasoning questions and answers. In order to increase the diversity of generated samples, it is supported to exclude already generated samples.
  
  See `example.py` for an example.