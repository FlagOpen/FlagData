"""
@misc{chen2023chinesewebtext,
      title={ChineseWebText: Large-scale High-quality Chinese Web Text Extracted with Effective Evaluation Model},
      author={Jianghao Chen and Pu Jian and Tengxiao Xi and Dongyi Yi and Qianlong Du and Chenglin Ding and Guibo Zhu and Chengqing Zong and Jinqiao Wang and Jiajun Zhang},
      year={2023},
      eprint={2311.01149},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""
# coding = utf-8
import os
import json
import jieba
import random
import argparse
import fasttext
from tqdm.auto import tqdm
from multiprocessing import Pool

jieba.setLogLevel(20)

stopwords = set([x.strip() for x in open(
    "./data/cn_stopwords.txt").readlines()])


def build(text):
    segs = jieba.lcut(text)
    segs = [x for x in segs if len(x) > 1 and x not in stopwords]
    return " ".join(segs)


def predict(input_file):
    file_dir, file_name = os.path.split(input_file)

    output_dir = os.path.join(file_dir, "fasttext")
    output_dir = output_dir.replace("data2", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name)

    lines = []
    with open(input_file, 'r', encoding='utf-8') as r_f:
        for line in r_f:
            lines.append(json.loads(line))

    # print("jieba cutting...")d
    seg_texts = [build(''.join(line["raw_content"])) for line in lines]

    # print("predicting...")
    labels, values = model.predict(seg_texts)

    # print("writing...")
    with open(output_file, 'w', encoding='utf-8') as w_f:
        for label, value, line in zip(labels, values, lines):
            _label = label[0].replace("__label__", "")
            _value = value[0] if value[0] <= 1 else 1
            line['fasttext_value'] = float(_value) if _label == 'clean' else float(1 - _value)

            w_f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dates', nargs='+', default=None)
    args = parser.parse_args()

    model = fasttext.load_model("models/model.bin")

    for clean_root in args.dates:

        file_name_list = [x for x in os.listdir(clean_root) if x.endswith(".jsonl")]
        print(file_name_list)

        with Pool(64) as p:
            for _ in tqdm(p.imap_unordered(predict,
                                           [os.path.join(clean_root, file_name) for file_name in file_name_list]),
                          total=len(file_name_list)):
                pass
