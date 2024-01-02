# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

import random
import re
from tqdm import tqdm
from utils import get_gpt4_response, clean_patterns
from prompt_templates import ability_extraction_prompt, sample_gen_ability_prompt, single_language_imitate_ques_prompt, \
    multi_language_imitate_ques_prompt, ceveat_precision_prompt, ceveat_unanswerable_prompt, ability_gen_prompt, \
    ability_gen_exclude_sample_prompt


def clean(response):
    ques = response.split('\n')
    ques = [re.sub(r'\d+\.\s+', '', i) for i in ques]
    ques = [re.sub(r'\d+\.', '', i) for i in ques]
    ques = [i.replace('\\', '') for i in ques]
    ques = [i.strip('\n') for i in ques]
    ques = [i for i in ques if len(i) > 5]
    return ques


class ImitateGenerator():
    def __init__(self, example_ls, additional_ceveat='', multi_language=False):
        self.example_ls = example_ls
        if multi_language:
            self.template_ques = single_language_imitate_ques_prompt
        else:
            self.template_ques = multi_language_imitate_ques_prompt
        self.template_ques = self.template_ques + '\n' + additional_ceveat
        self.multi_language = multi_language

    def gen_ques(self, num_samples, task, language, model='gpt4'):

        if self.multi_language:
            language_1 = language[0]
            language_2 = ', '.join(language[1:])
        else:
            language_1 = language_1 if isinstance(language, str) else language[0]
        ques_ls = []
        for i in tqdm(num_samples):
            n = random.randint(3, 5)
            sample_tmp = random.sample(self.example_ls, n)
            sample_str = '\n'.join(sample_tmp)
            if self.multi_language:
                template_ques = self.template_ques.format(task=task, sample_str=sample_str, language_1=language_1,
                                                          language_2=language_2)
            else:
                template_ques = self.template_ques.format(task=task, sample_str=sample_str, language_1=language_1)
            response = get_gpt4_response(template_ques, model)
            ques = clean(response)
            ques_ls.extend(ques)

        self.ques_ls = ques_ls
        return ques_ls

    def gen_ans(self, model='gpt4'):
        samples_gen = []
        num_fail = 0
        for ques in self.ques_ls:
            ans = get_gpt4_response(ques, model)

        if ans == '' or '抱歉' in ans:
            num_fail += 1
        else:
            ans = ans.strip('\n').strip()
            samples_gen.append([{'role': 'user', 'content': ques}, {'role': 'assistant', 'content': ans}])
        print(f"Total num: {str(len(self.ques_ls))}, Failed num: {str(num_fail)}")
        self.samples_gen = samples_gen
        return samples_gen


class AbilityExtractionGenerator():
    def __init__(self, example_ls, additional_ceveat=''):
        self.example_ls = example_ls

        self.template_ability_extraction = ability_extraction_prompt + '\n' + additional_ceveat[0]
        self.template_sample_gen = sample_gen_ability_prompt + '\n' + additional_ceveat[1]

    def ability_extraction(self, num_examples, model='gpt4'):

        ability_set = set()
        ability_ls = []
        for i in tqdm(num_examples):
            n = random.randint(3, 5)
            sample_tmp = random.sample(self.example_ls, n)
            sample_str = '\n'.join(sample_tmp)
            template_ability_extraction = self.template_ability_extraction.format(sample_str=sample_str)
            response = get_gpt4_response(template_ability_extraction, model)
            abilities = clean(response)

            for ability in abilities:
                ability_name = ability.split("：")[0].split(":")[0].strip()
                if ability_name not in ability_set:
                    ability_set.add(ability_name)
                ability_ls.append(ability)

        self.ability_ls = ability_ls
        return ability_ls

    def gen_sample(self, num_samples, model='gpt4'):
        samples_gen_ori = []
        num_fail = 0
        for i in tqdm(range(num_samples)):
            ability = random.sample(self.ability_ls, 1)[0]
            ability = ability.strip()
            indicator = random.random()
            if indicator < 0.2:
                ques_type = "选择"
            else:
                ques_type = "简答"

            indicator = random.random()
            if indicator < 0.15:
                difficulty = '偏难：4步以上推理才能得到答案'
            elif indicator < 0.6:
                difficulty = '适中：2-3步推理即可得到答案'
            else:
                difficulty = '简单：1-2步推理即可得到答案'
            template_sample_gen = self.template_sample_gen.format(ability=ability, ques_type=ques_type,
                                                                  difficulty=difficulty)

            sample = get_gpt4_response(template_sample_gen, model)
            samples_gen_ori.append(sample)

        samples_gen = []
        num_fail = 0
        patterns = ['题目：', '答案：']
        for sample in samples_gen_ori:
            try:
                ques, ans = sample['response'].split('###')
                ques = clean_patterns(ques.strip('\n'), patterns)
                ans = clean_patterns(ans.strip('\n'), patterns)
                samples_gen.append([{'role': 'user', 'content': ques}, {'role': 'assistant', 'content': ans}])
            except:
                num_fail += 1

        print(f"Total num: {str(len(self.ques_ls))}, Failed num: {str(num_fail)}")
        self.samples_gen = samples_gen
        return samples_gen


class AbilityDirectGenerator():
    def __init__(self, example_ls, additional_ceveat='', exclude_prev=True) -> None:
        self.example_ls = example_ls
        self.exclude_prev = exclude_prev

        self.ability_gen_exclude_prompt = ability_gen_exclude_sample_prompt + '\n' + additional_ceveat
        self.ability_gen_prompt = self.ability_gen_prompt + '\n' + additional_ceveat

    def gen_ques(self, num_examples, task, num_exclude=500, model='gpt4'):

        ques_ls = []
        for i in tqdm(range(num_examples)):
            if len(ques_ls) > num_exclude and self.exclude_prev:
                n = random.randint(3, 5)
                sample_tmp = random.sample(self.example_ls, n)
                sample_str = '\n'.join(sample_tmp)
                ability_gen_prompt = self.ability_gen_exclude_prompt.format(task=task, sample_str=sample_str)
            else:
                ability_gen_prompt = self.ability_gen_prompt.format(task=task)
            response = get_gpt4_response(ability_gen_prompt, model)
            ques = clean(response)
            ques_ls.extend(ques)

        return ques_ls

    def gen_ans(self, model='gpt4'):
        samples_gen = []
        num_fail = 0
        for ques in self.ques_ls:
            ans = get_gpt4_response(ques, model)

        if ans == '' or '抱歉' in ans:
            num_fail += 1
        else:
            ans = ans.strip('\n').strip()
            samples_gen.append([{'role': 'user', 'content': ques}, {'role': 'assistant', 'content': ans}])
        print(f"Total num: {str(len(self.ques_ls))}, Failed num: {str(num_fail)}")
        self.samples_gen = samples_gen
        return samples_gen
