from unstructured.partition.pdf import partition_pdf
import argparse


def pdf2txt(input_path, process_all):
    elements = partition_pdf(filename=input_path, infer_table_structure=True, strategy="hi_res")
    all_file = None

    if process_all:
        # 创建一个空字典，用于存储不同类别的元素列表
        tables = {}

        # 定义要筛选的类别列表
        categories = [
            "Table", "FigureCaption", "NarrativeText", "ListItem", "Title", "Address",
            "PageBreak", "Header", "Footer", "UncategorizedText", "Image", "Formula"
        ]

        # 遍历每个类别，筛选出对应类别的元素并存储到字典中
        for category in categories:
            # 检查是否为 "Table" 类别，需要分别处理 "table.metadata.text_as_html"
            if category == "Table":
                tables[category] = [el for el in elements if el.category == category]
                # 另外存储 "table.metadata.text_as_html"
                tables["Table_text_as_html"] = [el.table.metadata.text_as_html for el in elements if
                                                el.category == category]
            else:
                # 对于其他类别，直接存储筛选的元素
                tables[category] = [el for el in elements if el.category == category]
        all_file = tables
    else:
        print(elements)
        all_file = "\n\n".join([str(el) for el in elements])

    return all_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pdf2txt")
    parser.add_argument("--input", "-i", type=str, required=True, help="input Catalogue path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")

    # 修改 --all 参数为布尔类型，并更改变量名以避免与内置函数冲突
    parser.add_argument("--process_all", action="store_true",
                        help="process all articles if specified, otherwise split by category list, default is all")

    args = parser.parse_args()
    result = pdf2txt(args.input, args.process_all)

    # 将结果保存到文件
    output_file_path = args.output  # 指定输出文件路径，可以根据实际需求修改
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        if isinstance(result, dict):
            # 如果结果是字典类型，将字典内容写入文件
            for key, value in result.items():
                output_file.write(f"Category: {key}\n")
                output_file.write(f"{value}\n\n")
        else:
            # 如果结果不是字典，直接写入文件
            output_file.write(result)
