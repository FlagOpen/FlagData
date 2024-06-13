import re
from flagdata.cleaner.base_cleaner import Cleaner
from typing import Dict
import json
import logging
from tqdm import tqdm


class ArxivCleaner(Cleaner):
    def __init__(self, cleaning_steps):
        super().__init__(cleaning_steps)
        # 可以添加更多Arxiv特有的初始化逻辑

    def clean(self):
        print("arxiv_cleaner start......")
        with open(self.input_path, "r", encoding="utf8") as fr:
            try:
                for line in tqdm(fr.readlines()):
                    text = json.loads(line.strip())
                    source_text = text[self.source_key]
                    for func_name in self.config.get('ArxivCleaner', []):
                        if hasattr(self, func_name):
                            result_text = getattr(self, func_name)(source_text)
                            text[self.result_key] = result_text
                            source_text = result_text
                    self.write_jsonl_file(text)
            except Exception as e:
                logging.warning("read error: ", e)

    # 实现 Arxiv 清洗逻辑
    def _clean_tex_file(self, file_content: str, arg_macros: Dict, non_arg_macros: Dict) -> str:
        pattern = r"^(.*?)("
        pattern += r"\\\bchapter\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bpart\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bsubsubsection\b\*?(?:\[(.*?)\])?\{(.*?)\}|"
        pattern += r"\\\bparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        pattern += r"\\\bsubparagraph\b\*?(?:\[(.*?)\])?\{(.*?)\}"
        pattern += r")"

        if not re.search(pattern, file_content, flags=re.DOTALL):
            return ""

        file_content = re.sub(pattern=pattern, repl=r"\2", string=file_content, flags=re.DOTALL)

        file_content = re.sub(pattern=r"(?m)^%.*\n?", repl=r"", string=file_content, flags=re.MULTILINE)

        file_content = re.sub(pattern=r"[^\\]%.+$", repl=r"", string=file_content, flags=re.MULTILINE)

        pattern = r"("
        pattern += r"\\appendix|"
        pattern += r"\\begin\{references\}|"
        pattern += r"\\begin\{REFERENCES\}|"
        pattern += r"\\begin\{thebibliography\}|"
        pattern += r"\\bibliography\{.*\}"
        pattern += r").*$"

        file_content = re.sub(pattern=pattern, repl=r'', string=file_content, flags=re.DOTALL)

        for macro_name, macro_value in non_arg_macros.items():
            macro_name_escaped = re.escape(macro_name)

            try:
                file_content = re.sub(pattern=r"(" + macro_name_escaped + r")" + r"([^a-zA-Z0-9])",
                                      repl=macro_value + r"\2", string=file_content)
            except re.error as e:
                print("Error occurred while processing:", e)
                print("Problematic content:", macro_name_escaped)

            # file_content = re.sub(pattern=r"(" + macro_name + r")" + r"([^a-zA-Z0-9])",
            #                       repl=macro_value + r"\2", string=file_content)

        for macro_name, macro_value in arg_macros.items():
            pass

        return file_content

    def _build_non_arg_macros_dict(self, file_content: str) -> Dict[str, str]:
        non_arg_nc_reg = re.compile(
            pattern=r'\\\bnewcommand\b\*?\{(\\[a-zA-Z0-9]+?)\}\{(.*?)\}$',
            flags=re.MULTILINE
        )

        non_arg_def_reg = re.compile(
            pattern=r'\\def\s*(\\[a-zA-Z0-9]+?)\s*\{(.*?)\}$',
            flags=re.MULTILINE
        )

        macros = {}
        for reg in [non_arg_nc_reg, non_arg_def_reg]:
            for match in reg.finditer(file_content):
                macro_name = match.group(1).encode("unicode-escape").decode("utf-8")
                macro_val = match.group(2).encode("unicode-escape").decode("utf-8")
                macros[macro_name] = macro_val

        return macros

    def process_text(self, text):
        non_arg_macros = self._build_non_arg_macros_dict(text)
        return self._clean_tex_file(text, {}, non_arg_macros)
