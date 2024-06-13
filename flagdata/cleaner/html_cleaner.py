from flagdata.cleaner.utils.common_utils import end_clip, remove_specific_patterns, remove_control_chars, \
    remove_extraspace, remove_unwanted_lines, drop_docs_exceeding_newline_proportion, drop_doc_below_ratio
from flagdata.cleaner.utils.extractor import ExtractorPipeline, ContentExtractor, TimeExtractor, TitleExtractor


def predict(output_file, content):
    with open(output_file, 'w', encoding='utf-8') as w_f:
        try:
            title_extractor = TitleExtractor()
            time_extractor = TimeExtractor()
            content_extractor = ContentExtractor()
            # 创建 ExtractorPipeline 实例并传入提取器列表
            pipeline = ExtractorPipeline(title_extractor, time_extractor, content_extractor)
            extracted_items, content = pipeline.extract(content)
        except Exception as e:
            print(f"Error decoding JSON: {e}")
        if drop_docs_exceeding_newline_proportion(content):
            return
        if drop_doc_below_ratio(content):
            return
        content = end_clip(content)
        content = remove_specific_patterns(content)

        content = remove_control_chars(content)
        content = remove_extraspace(content)
        content = remove_unwanted_lines(content)
        w_f.write(content[1:])


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == '__main__':
    output_file = 'output/html_demo_output.txt'
    input_file = 'input/html_demo_input.txt'
    content = read_file(input_file)
    predict(output_file, content)
