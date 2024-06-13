from flagdata.cleaner.base_cleaner import Cleaner
import yaml
import importlib

from flagdata.cleaner.utils.string_utils import is_camel_case, camel_to_snake


class CleanerBuilder:
    @staticmethod
    def build(cleaning_steps_yaml: str) -> Cleaner:

        # 加载YAML配置文件
        with open(cleaning_steps_yaml, 'r') as file:
            cleaning_steps_config = yaml.safe_load(file)
        # 获取配置文件中指定的清洗器类名
        cleaner_class_name = cleaning_steps_config["basic"].get("cleaner_class")

        if is_camel_case(cleaner_class_name):
            cleaner_class_name_path = 'flagdata.cleaner_v2.' + camel_to_snake(cleaner_class_name)

        else:
            raise ValueError(f"{cleaner_class_name} is not CamelCase")
        # 创建清洗器实例
        try:
            module = importlib.import_module(cleaner_class_name_path)
        except ImportError:
            print(f"Module {cleaner_class_name} not found.")
            return None
        try:
            cleaner_class = getattr(module, cleaner_class_name)

            # Instantiate the class and call its clean method
            cleaner_instance = cleaner_class(cleaning_steps_yaml)
        except:
            print("Cleaner class not specified in the configuration file")
            return None
        # 对象初始化异常不捕获，外层处理
        return cleaner_instance


if __name__ == '__main__':
    # cleaning_steps_yaml = 'configs/default_clean.yaml'
    cleaning_steps_yaml = 'configs/arxiv_clean.yaml'
    # cleaning_steps_yaml = 'configs/text_clean.yaml'
    CleanerBuilder.build(cleaning_steps_yaml).clean()
