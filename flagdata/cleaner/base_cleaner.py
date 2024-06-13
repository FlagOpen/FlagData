import yaml
from abc import ABC, abstractmethod
import logging
import json


class Cleaner(ABC):
    """
    清洗类抽象基类
    1.每个子类需要重写__init__、clean 两个函数，子类有相同逻辑的可以调用父类方法
    2.run()方法无需重写，run()负责调度 clean、dedup、language_identification，供外部主逻辑调用
    3.异常处理：统一抛出去，在主干流程内进行捕获并触发机器人报警、打错误日志

    后续存在一些公共方法，再提到此处
    """

    @staticmethod
    def _read_config(config_path: str):
        with open(config_path, "r", encoding="utf8") as fr:
            return yaml.safe_load(fr)

    def __init__(self, config_path="configs/default_clean.yaml"):
        """
        初始化清洗器，接收一个清洗步骤的配置。
        """
        # 初始化一个空列表用于存储值为 True 的键
        self.config = self._read_config(config_path)
        logging.info(self.config)
        print(self.config)
        self.input_path = self.config["basic"].get("input")
        self.output_path = self.config["basic"].get("output")
        self.source_key = self.config['basic'].get("source_key")
        self.result_key = self.config['basic'].get("result_key")

    @abstractmethod
    def clean(self):
        pass

    """
    添加子类需要实现自己的clean
    """

    def run(self):
        # todo：串联clean、dedup、language_identification
        pass

    def read_jsonl_file(self):
        with open(self.input_path, "r", encoding="utf8") as fr:
            try:
                for line in fr:
                    text = json.loads(line.strip())
            except Exception as e:
                logging.warning("read error: ", e)
            return text

    def write_jsonl_file(self, text: str):
        """
        写
        :return:
        """
        with open(self.output_path, "a", encoding="utf8") as fw:
            try:
                fw.write(json.dumps(text, ensure_ascii=False) + "\n")
            except Exception as e:
                logging.warning("write error: ", e)


if __name__ == '__main__':
    cleaner = Cleaner()
