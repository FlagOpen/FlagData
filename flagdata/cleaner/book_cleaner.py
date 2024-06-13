import re, time
import logging, os
import json
import chardet
import langdetect
from functools import partial
import shutil
from shutil import copy
from collections import defaultdict
import traceback
import math
import zipfile
from bs4 import BeautifulSoup
from langdetect import detect
import subprocess

from flagdata.cleaner.utils.time_formatter import timeout


class BookCleaner():
    def __init__(self, config_path, workspace, target_path):
        super().__init__()
        base_workspace = os.path.join(workspace, 'book_temp_dir')

        temp_index = 1
        self.workspace = base_workspace + f'{temp_index}'
        while os.path.exists(self.workspace):
            temp_index += 1
            self.workspace = base_workspace + f'{temp_index}'

        os.makedirs(self.workspace)
        logging.info(f"Created directory: {self.workspace}")

        self.target_path = target_path
        self.save_index = 0
        try:
            with open(config_path, 'r') as config_file:
                book_config = json.load(config_file)
            self.KEYWORDS_SET = book_config["KEYWORDS_SET"]
            self.KEYWORDS_BY_PARTS = book_config["KEYWORDS_BY_PARTS"]
        except Exception as e:
            logging.error(f"load book_config failed: {str(e)}")

    @timeout(120)
    def convert_ebook(self, input_file, converted_file):
        try:
            command = ['ebook-convert', input_file, converted_file]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"Error during conversion: {e}")

    # check input file fomart: epub/azw/mobi
    def convert_file_2epub(self, input_path, book_name):
        if not (input_path.endswith('.epub') or input_path.endswith('.azw') or input_path.endswith(
                '.azw3') or input_path.endswith('.mobi')):
            logging.error(f"Invalid file extension for file: {input_path}. Expected .epub, .azw, .azw3 , or .mobi")
            raise ValueError("Invalid file extension. The file must be an .epub, .azw, .azw3 or .mobi file.")

        if input_path.endswith('.azw') or input_path.endswith('.azw3') or input_path.endswith('.mobi'):
            # Convert to epub file and store it in workspace
            epub_path = os.path.join(self.workspace, f'{book_name}.epub')

            try:
                if not os.path.exists(epub_path):
                    # Redirect standard output and standard error to DEVNULL
                    subprocess.run(['ebook-convert', input_path, epub_path],
                                   check=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   timeout=300)
                else:
                    logging.info(f"{epub_path} already exists.")
            except Exception as e:
                logging.error(f"Error during conversion: {str(e)}")
        else:
            epub_path = input_path

        return epub_path

    def unzip_epub(self, zip_path, unzip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def zip_epub(self, unzip_path, epub_path):
        with zipfile.ZipFile(epub_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(unzip_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, unzip_path))

    def find_html_files(self, path):
        html_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.html') or file.endswith('.xhtml'):
                    html_files.append(os.path.join(root, file))
        return html_files

    def process_html_file(self, html_file):
        try:
            soup = BeautifulSoup(open(html_file, 'r'), 'html.parser')
        except:
            encoding = self.detect_encoding(html_file)
            soup = BeautifulSoup(open(html_file, 'r', encoding=encoding, errors='ignore'), 'html.parser')

            # Rule 1, if there is a picture and a paragraph in the div, delete the paragraph (delete the caption of the picture)
        for div in soup.find_all('div'):
            if div.find('img') and len(div.find_all('p')) == 1:
                logging.info(f'{str(div)} is deleted form epub file')
                div.find('p').decompose()

        '''
        Rule 2, if total text length in <span> >= that in <p>, It is considered that this page is most likely not the main 
        text content. By deleting the text in the span, the directory, copyright, cover, author introduction and other meta 
        information are deleted (using 3000 as the threshold to exclude the case where the main text is directly written in
         <span>)
        '''

        total_span_text_length = sum(len(span.get_text()) for span in soup.find_all('span'))
        total_p_text_length = sum(len(p.get_text()) for p in soup.find_all('p'))
        if total_span_text_length >= total_p_text_length:
            if total_span_text_length < 3000:
                for span in soup.find_all('span'):
                    print(span.get_text())
                for span in soup.find_all('span'):
                    span.decompose()
        return

    def process_epub(self, input_path):
        workspace = self.workspace
        zip_path = os.path.join(workspace, 'temp.zip')
        unzip_path = os.path.join(workspace, 'temp')
        epub_path = os.path.join(workspace, 'temp.epub')

        try:
            if os.path.exists(unzip_path):
                shutil.rmtree(unzip_path)
                logging.info(f"Directory {unzip_path} has been removed.")

            shutil.copy(input_path, zip_path)
            logging.info(f"File from {input_path} has been copied to {zip_path}.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

        os.makedirs(unzip_path, exist_ok=True)

        self.unzip_epub(zip_path, unzip_path)
        html_files = self.find_html_files(unzip_path)

        for html_file in html_files:
            self.process_html_file(html_file)

        self.zip_epub(unzip_path, epub_path)

        return epub_path

    def detect_encoding(self, file_path):
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        return chardet.detect(raw_data)['encoding']

    def content_length_lower_ratio(self, lines, threshold):
        lens = [self.count_characters(line) for line in lines if self.count_characters(line) != 0]
        low_count = sum(1 for x in lens if x <= threshold)
        if len(lens) != 0:
            return low_count / len(lens)
        else:
            return 0

    def count_characters(self, text):
        # Calculate the number of Chinese characters
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        # Count the number of English words
        english_words = re.findall(r'\b[a-zA-Z]+\b', text)
        english_count = len(english_words)
        return chinese_count + english_count

    def remove_excess_newlines(self, content):
        return re.sub(r'\n{3,}', '\n\n', content)

    def remove_consecutive_stars(self, content):
        return re.sub(r'\*{2,}', '', content)

    def check_part(self, parts):

        def has_keyword(lines, keywords, max_count):
            for line in lines:
                if any(keyword in line for keyword in keywords) and self.count_characters(line) < max_count:
                    return True
            return False

        def evaluate_part(part):
            lines = part.split('\n')
            checks = []
            for keyword in self.KEYWORDS_BY_PARTS:
                checks.append(has_keyword(lines, keyword["keywords"], keyword["threshold"]))
            if self.content_length_lower_ratio(lines, 100) != 1:
                checks.append(False)
            elif self.content_length_lower_ratio(lines, 40) >= 0.8:
                checks.append(True)
            else:
                checks.append(False)

            # Only parts with less than 5000 characters can be deleted
            return any(checks) and self.count_characters(part) <= 5000

        evaluate_results = [evaluate_part(part) for part in parts]

        total_length = sum(len(part.split('\n')) for part in parts)
        # Calculate the cumulative length of each part to determine its position in the total text
        cumulative_length = 0
        length_results = []

        for part in parts:
            if cumulative_length + len(
                    part.split('\n')) <= total_length * 0.2 or cumulative_length >= total_length * 0.8:
                length_results.append(True)
            else:
                length_results.append(False)
            cumulative_length += len(part.split('\n'))

        should_delete_ls = [evaluate_result and length_result for evaluate_result, length_result in
                            zip(evaluate_results, length_results)]
        parts = [part for part, should_delete in zip(parts, should_delete_ls) if not should_delete]

        return parts

    def pre_process_zh(self, content, lines):
        parts = re.split(r'\n{4,}', content)
        parts = self.check_part(parts)
        content = '\n\n'.join(parts)
        content = self.remove_excess_newlines(content)
        content = self.remove_consecutive_stars(content)
        lines = content.split('\n')

        return lines

    def clean_by_lines(self, lines, keywords_set):
        def apply_rules(keyword, line, index, total_lines):
            # 只有前 ratio_s %和后 ratio_e% 的行会被删除
            key, length, is_lower, ratio_s, ratio_e = keyword
            tar_line = line.lower().strip() if is_lower else line.strip()
            key = key.lower() if is_lower else key

            if (length != 0 and len(line) < length and key in tar_line) or (tar_line == key):
                start_index = index if index < total_lines * ratio_s else None
                end_index = index if index >= math.ceil(total_lines * (1 - ratio_e)) else None
                return start_index, end_index
            return (None, None)

        figure_pattern = re.compile(r"^(图|表|Figure|Table)\s*\d+(-\d+)?(\.\d+)?\s*[\u4e00-\u9fa5A-Za-z]*$",
                                    re.IGNORECASE)

        lines = [line for line in lines if not figure_pattern.fullmatch(line)]

        del_keys = ['ePUBw.COM', '电子书下载']
        lines = [line for line in lines if not any(keyword in line for keyword in del_keys)]

        total_lines = len(lines)

        early_index = None
        late_index = None

        for index, line in enumerate(lines):
            lower_line = line.lower()
            line_len = self.count_characters(line)

            # Check for lines starting with a specific start
            if 'chapter' in lower_line:
                digits = lower_line[lower_line.find('chapter') + len('chapter'):]
                matches = re.search(r'\d[\d\s]*\d|\d', digits)

                if matches:  # Check if a number was found
                    digits = re.sub(r'\s+', '', matches.group(0))
                else:
                    digits = ''  # If no digits are found, set digits to an empty string
                cleaned_text = re.sub(r'[^a-zA-Z]', '', lower_line)

                if digits == '1' or cleaned_text == 'chapterone':  # If "1" is in the number
                    # Determine whether the number of rows in this row is in the top 30% of the total number of rows
                    if index < total_lines * 0.3 and line_len < 10:
                        early_index = index

            if line.strip() in ['I .', 'I.', 'I', ] or lower_line.strip() == 'chapter i' or 'chapter i ' in lower_line:
                if index < total_lines * 0.3 and line_len < 20:
                    early_index = index

            for keyword in keywords_set:
                early_candidate, late_candidate = apply_rules(keyword, line, index, total_lines)

                if early_candidate is not None and (early_index is None or early_candidate + 1 > early_index):
                    early_index = early_candidate + 1

                if late_index is None and late_candidate is not None:
                    late_index = late_candidate

        if early_index == None:
            early_index = 0
        if late_index == None:
            late_index = total_lines

        lines = lines[early_index: late_index]

        lines_to_delete = set()  # Store the index of the row to be deleted
        lines_length = [self.count_characters(line) for line in lines]
        # consecutive
        consecutive_count = 0  # The number of rows that meet the conditions
        temp = set()

        for i, line in enumerate(lines):
            if lines_length[i] < 100 and (('第' in line and '章' in line) or ('第' in line and '幕' in line) or (
                    'chapter' in line.lower()) or re.match(r'^\d+\.', line.strip()) or ('............' in line)):
                # Increase the consecutive row count
                consecutive_count += 1
                temp.add(i)
                if consecutive_count >= 3:
                    lines_to_delete.update(temp)
            elif line in ['', '\n']:  #
                consecutive_count += 0
                temp.add(i)
            else:
                consecutive_count = 0
                temp.clear()

            if lines_length[i] < 50 and (
                    '.com' in line.lower() or 'www.' in line.lower() or '@' in line.lower() or line[:2] == '##'):
                lines_to_delete.add(i)

        # Generate a new list, excluding the rows to be deleted, but keeping the empty rows
        lines = [line for i, line in enumerate(lines) if i not in lines_to_delete]
        return lines

    def short_line_ratio(self, lines, length_threshold):
        count_less = 0
        count_more = 0

        for line in lines:
            line = line.strip()
            num_characters = self.count_characters(line)
            # Check the end character and number of Chinese characters in the line to classify the count
            if num_characters > length_threshold:
                count_more += 1
            else:
                count_less += 1

        # Calculate the percentage of lines with less than 50 Chinese characters
        total_lines = len(lines)
        ratio = count_less / total_lines if total_lines > 0 else 1
        if count_more > 20:
            return 0
        else:
            return ratio

    def write_jsonl_file(self, content):
        with open(self.target_path, 'a') as output_file:
            line = json.dumps({"content": content}, ensure_ascii=False)
            output_file.write(line + '\n')

            # with open(self.target_path + f'/{self.save_index}.txt', 'w') as output_file:
        #     output_file.write(content)
        # print(self.save_index)
        self.save_index += 1

    def clean(self, input_path):
        # 清洗前清空 workspace
        workspace = self.workspace

        if os.path.exists(workspace):
            shutil.rmtree(workspace)
            print(f"Directory removed: {workspace}")
        os.makedirs(workspace)
        print("book_cleaner start......")

        book_name = os.path.basename(input_path)
        book_name, _ = os.path.splitext(book_name)
        out_txt_path = os.path.join(workspace, f'{book_name}.txt')

        try:
            input_path = self.convert_file_2epub(input_path, book_name)
        except Exception as e:
            logging.error("convert_file_2epub error: ", e)
            return None

        try:
            epub_path = self.process_epub(input_path)
        except Exception as e:
            logging.error("process_epub error: ", e)
            return None

        try:
            self.convert_ebook(epub_path, out_txt_path)
        except Exception as e:
            logging.warning("convert_ebook failed: ", e)
            return None

        try:
            encoding = self.detect_encoding(out_txt_path)
            with open(out_txt_path, 'r', encoding=encoding) as file:
                content = file.read()
                language = langdetect.detect(content)
                file.seek(0)
                lines = file.readlines()
        except Exception as e:
            logging.warning("open_ebook failed: ", e)
            return None

        if not (language == 'zh-cn' or language == 'en'):
            logging.info("Languages other than zh-cn and en")
            return None
        else:
            if language == 'zh-cn' and self.short_line_ratio(lines, 100) > 0.9:
                logging.info("not enough long lines")
                return None
            elif language == 'en' and self.short_line_ratio(lines, 150) > 0.9:
                logging.info("not enough long lines")
                return None
            else:
                lines = self.pre_process_zh(content, lines) if language == 'zh-cn' else lines
                lines = self.clean_by_lines(lines, self.KEYWORDS_SET)

                content = '\n'.join(lines)
                self.write_jsonl_file(content)


if __name__ == '__main__':
    test_directory = 'input/book_demo_data'
    target_path = 'output/book_demo_output.jsonl'

    config_path = 'configs/book_config.json'

    book_cleaner = BookCleaner(config_path=config_path,
                               workspace='input/book_demo_temp',
                               target_path=target_path)

    books = [os.path.join(test_directory, filename) for filename in os.listdir(test_directory)]
    for book_path in books:
        book_cleaner.clean(book_path)
