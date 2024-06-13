import re
import json
import yaml
import unicodedata
import string
from tqdm import tqdm


class TextCleaner():
    def __init__(self, config_path="config.yaml"):
        self.config = self._read_config(config_path)

    @staticmethod
    def _read_config(config_path: str):
        with open(config_path, "r", encoding="utf8") as fr:
            return yaml.safe_load(fr)

    def clean(self):
        print("html_cleaner start......")
        with open(self.config["basic"].get("input"), "r", encoding="utf8") as fr, open(self.config["basic"].get("output"), "w", encoding="utf8") as w:
            try:
                for line in tqdm(fr.readlines()):
                    line = json.loads(line)
                 
                    text = line[self.config["basic"].get("source_key")]
                    if self.drop_docs_exceeding_newline_proportion(text):
                        continue
                    if self.drop_doc_below_ratio(text):
                        continue
                    text = self.end_clip(text)
                    text = self.remove_specific_patterns(text)
                    
                    text = self.remove_control_chars(text)
                    text = self.remove_extraspace(text)
                    text = self.remove_unwanted_lines(text)
                    line[self.config["basic"].get("result_key")] = text
                    w.write(json.dumps(line, ensure_ascii=False)+"\n")
                    
            except Exception as e:
                logging.warning("read error: ", e)


    def remove_control_chars(self, text):
        return ''.join(ch for ch in text if (unicodedata.category(ch) != 'Cc' or ch == '\n'))  # 

    def remove_extraspace(self, text):
        normalized_string = re.sub(r'[\xa0\u3000]+', ' ', text)
        normalized_string = re.sub(r'[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F-\x9F]+', '', normalized_string)
        return normalized_string

    def end_clip(self, text):
        chinese_punctuation = "。！？"
        english_punctuation = ".?!"
        all_punctuation = chinese_punctuation + english_punctuation
        if text[-1] in all_punctuation:
            return text
        else:
            match = re.search(r'[{}]'.format(re.escape(all_punctuation)), text[::-1])
            if match:
                return text[:len(text) - match.start()]
            else:
                return ""

    # 删除含有特定内容的句子
    def remove_specific_patterns(self, text):
        for pattern in self.config["basic"].get("pattern"):
            text = re.sub(r'^.*' + re.escape(pattern) + r'.*$\n?', '', text, flags=re.MULTILINE)
        return text.strip()

    # 删除多行空格
    def remove_unwanted_lines(self, text):
        return re.sub(r'\n\n+', '\n', text)

    # 删除数字比率大于0.5的文档
    def drop_doc_below_ratio(self, text, ratio=0.5):
        def count_characters(doc):
            chinese_chars = re.findall(r'[\u4e00-\u9fff]+', doc)
            chinese_count = sum(len(char) for char in chinese_chars)
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            word_count = len(words)
            return chinese_count + word_count
            
        numbers = re.findall(r'\d+\.\d+|\d+', text)
    
        if count_characters(text) != 0:
            score = len(numbers) / count_characters(text)
            if score > ratio:
                return True
            else:
                return False
        else:
            return False


    def drop_docs_exceeding_newline_proportion(self, text, max_newline_proportion=0.25):
        def count_newlines(doc):
            return doc.count('\n')
        
        def count_characters(doc):
            chinese_chars = re.findall(r'[\u4e00-\u9fff]+', doc)
            chinese_count = sum(len(char) for char in chinese_chars)
            words = re.findall(r'\b[a-zA-Z]+\b', text)
            word_count = len(words)
            return chinese_count + word_count
        
        newline_count = count_newlines(text)
        char_count = count_characters(text)
    
        if char_count != 0:
            newline_proportion = newline_count / char_count
            if newline_proportion <= max_newline_proportion:
                return False
            else:
                return True
        else:
            return False


if __name__ == '__main__':
    html_clean = HTMLCleaner("textcleaner_config.yaml")
    html_clean.clean()
