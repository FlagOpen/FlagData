# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

from strategy import *
from utils import *
import pandas as pd
import random
import copy
import sys

sys.path.append('')
from multiprocess_call_api import run_multiprocess, get_gpt4_response
import timeout_decorator
import openai
import json
import re
import os

sample_ls = []

TIMEOUT = 240

openai.api_key = ''  # os.getenv("AZURE_OPENAI_KEY")
openai.api_base = ''  # os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = ''
openai.api_version = ''  # this may change in the future

deployment_name = ''  # This will correspond to the custom name you chose for your deployment when you deployed a model.

wkdir = ""
output_path = 'generated_dat.jsonl'

example_ls = [
    '我小时候吃虾，然后出现过敏反应。几年后，我吃了带有章鱼的虾，然后出现了更严重的过敏反应。每次我去那些有强烈虾味的餐馆，我都会一贯性地出现过敏反应。这可以推断出：',
    '小明观察到他周围的人们都喜欢吃巧克力。因此，他得出结论：所有人都喜欢吃巧克力。请问这个结论是否可靠？如果可靠，可以如何解释这种观察结果？如果不可靠，应该如何修正这个结论？',
    '在一个实验中，A组实验者使用药物X，B组实验者使用药物Y。最后发现A组的痊愈率明显高于B组。基于这一结果，我们可以得出什么结论？',
    '如果一个人经常迟到，那么他的时间观念是强还是弱？',
    '如果一个地区的犯罪率降低了，那么这个地区的治安是提高了还是降低了？']

task = '逻辑推理'

generator = ImitateGenerator(example_ls, additional_ceveat='', multi_language=False)
generator.gen_ques(num_samples=100, task=task, language='中文', model='gpt4')
generator.gen_ans()

generator = AbilityExtractionGenerator(example_ls)
generator.gen_ques(num_examples=100)
generator.gen_ans()

generator = AbilityDirectGenerator(example_ls, additional_ceveat='', exclude_prev=True)
generator.gen_ques(num_examples=100, task=task, num_exclude=500, model='gpt4')
generator.gen_ans()

out_path = os.path.join(wkdir, output_path)
os.makedirs(out_path, exist_ok=False)
save_jsonl(generator.samples_gen, output_path)
