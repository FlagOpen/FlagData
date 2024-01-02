import os
import requests
import json, copy
import openai

openai.api_key = ''  # os.getenv("AZURE_OPENAI_KEY")
openai.api_base = ''  # os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = ''
openai.api_version = ''  # this may change in the future

deployment_name = 'llmsft'  # This will correspond to the custom name you chose for your deployment when you deployed a model.


def get_chatgpt_response(query):
    # Send a completion call to generate an answer
    response = openai.ChatCompletion.create(
        engine=deployment_name,  # engine = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    if 'content' not in response['choices'][0]['message']:
        raise Exception(f'No content in response: {query}')
    response = response['choices'][0]['message']['content']
    return response


# ----------------------------------------------------------------------------------
level1 = {'理解能力': "考察语言模型对自然语言的理解。",
          '知识运用': "考察语言模型对通用知识或专业知识的应用能力。",
          '推理能力': "考察语言模型对数学，符号，事实，故事情节等内容的推理能力。",
          '特殊生成': "考察语言模型生成文本的能力，文本需要满足指令的风格，内容，语气等要求。",
          '解释能力': "考察语言模型的对文段，常识的理解。",
          '价值观': "考察语言模型的价值观。",
          '幻觉': "是否会生成不真实的信息。",
          '通用综合能力': "多学科的综合能力，对标人类高中毕业水平，如多学科高考，SAT考试。",
          '领域专家能力': "专业领域的综合能力，对标人类某个领域职业资格的水品，如律师、医师、教师资格考试等。"}

level2_1 = {'文化理解': '对于文化的理解，特别是对中国文化的理解，如谜语、歇后语、诗歌、对联等',
            '信息分析': '对文本中包含的信息分析能力，如分类（情感、领域），语义匹配，信息检索',
            '信息提取': '按照要求对给定文本中的关键信息进行抽取的能力，如关键词识别，代词消歧义，个人信息识别等',
            '信息概括': '按照要求对给文本进行总结概括的能力，如摘要，概述等',
            '跨语言理解': '理解多语言的能力，如翻译、多语言任务等',
            '判别评价': '按照要求对给定文本进行判别评价的能力，如作文批改，语法/文法纠错'}
level2_2 = {'知识问答': '通过问答的形式，考察对知识的理解和运用的能力',
            '常识问答': '通过问答的形式，考察对常识的掌握、分析和运用的能力',
            '事实问答': '通过问答的形式，考察对事实的掌握'}
level2_3 = {'知识推理': '利用知识的逻辑关系来推理和证明从而得出结论的能力',
            '符号推理': '进行抽象符号的推理'}
level2_4 = {'条件生成': '在给定限制下，生成文本的能力，如完形填空，信件生成',
            '代码生成': '按照要求生成代码的能力',
            '创意生成': '给定场景，完成开放性、有创意性的文本'}
level2_5 = {'心理疏导': '对于一些心理问题进行解答和疏导',
            '语言解析': '对给定文本进行解释的能力'}
level2_6 = {'歧视偏见': '是否会生成具有偏见和歧视性的信息，如对种族、民族、信仰、国别、地域、性别、年龄、职业等的偏见',
            '安全': '是否会生成涉及到违法犯罪，敏感话题，反面诱导，隐私财产等安全方面的信息',
            '伦理道德': '生成的内容是否符合人类的伦理道德'}


def find_in_keys(choice, rsp1):
    task_list = []
    for word in choice.keys():
        if word in rsp1:
            task_list.append(word)

    if task_list != []:
        return task_list[0]
    else:
        return None


def temp_level1(input):
    return f"{level1}\n\n上面描述了人工智能模型需要具备的多种能力。\n请根据上述信息，判断这个指令<{input}>最能考察到上述哪种能力。请从上述的能力列表中挑选出来一个输出，即使不属于任何一类，找一类以上列表中相近的输出。"


def temp_level2(input, level1):
    tasks_desc = {}

    if level1 in ['幻觉', '通用综合能力', '领域专家能力']:
        return "没有能力等级2"
    elif level1 == "理解能力":
        tasks_desc = level2_1
    elif level1 == '知识运用':
        tasks_desc = level2_2
    elif level1 == "推理能力":
        tasks_desc = level2_3
    elif level1 == "特殊生成":
        tasks_desc = level2_4
    elif level1 == "解释能力":
        tasks_desc = level2_5
    elif level1 == "价值观":
        tasks_desc = level2_6
    else:
        return None

    return f"{tasks_desc}\n\n上面描述了人工智能模型需要具备的多种能力。\n请根据上述信息，判断这个指令<{input}>最能考察到上述哪种能力。请从上述的能力列表中挑选出来一个输出，即使不属于任何一类，找一类以上列表中相近的输出。", tasks_desc


# ----------------------------------------------------------------------------------


def test(input):
    instruct1 = temp_level1(input)
    rsp1 = get_chatgpt_response(instruct1)
    temp_res = {}
    temp_res["问题"] = copy.deepcopy(input)
    temp_res["回复1"] = copy.deepcopy(rsp1)
    level1_ans = find_in_keys(level1, rsp1)

    if level1_ans:
        instruct2, tasks_desc = temp_level2(input, level1_ans)
        rsp2 = get_chatgpt_response(instruct2)
        temp_res["回复2"] = copy.deepcopy(rsp2)
        level2_ans = find_in_keys(tasks_desc, rsp2)

        if level2_ans == "没有能力等级2":
            temp_res["能力等级1"] = level1_ans
            return temp_res

        elif level2_ans:
            temp_res["能力等级1"] = level1_ans
            temp_res["能力等级2"] = level2_ans
            return temp_res


def multiTest(input_list):
    res = []
    for item in input_list:
        res.append(test(item))

    keys_level1 = list(level1.keys())
    capacity_level1 = {}
    for key in keys_level1:
        capacity_level1[key] = 0
    for item in res:
        if "能力等级1" in item:
            if item["能力等级1"] not in capacity_level1:
                capacity_level1[item["能力等级1"]] = 1
            else:
                capacity_level1[item["能力等级1"]] += 1

    keys_level2 = list(level2_1.keys()) + list(level2_2.keys()) + list(level2_3.keys()) + list(level2_4.keys()) + list(
        level2_5.keys()) + list(level2_6.keys()) + ['伦理道德', '幻觉', '通用综合能力', '领域专家能力']
    capacity_level2 = {}
    for key in keys_level2:
        capacity_level2[key] = 0
    for item in res:
        if "能力等级2" in item:
            if item["能力等级2"] not in capacity_level2:
                capacity_level2[item["能力等级2"]] = 1
            else:
                capacity_level2[item["能力等级2"]] += 1

    return capacity_level1, capacity_level2
    # 返回两个能力等级的统计结果


if __name__ == '__main__':
    # text = "端午节有哪些习俗？"
    # print(test(text))

    text = ["端午节有哪些习俗？", "解释一下勾股定理"]
    print(multiTest(text))
