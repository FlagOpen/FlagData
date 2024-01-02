# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import json

import argparse
from unstructured.partition.epub import partition_epub


def epub2txt(input_path, process_all):
    elements = partition_epub(filename=input_path)
    if process_all:
        # Create an empty dictionary to store lists of elements in different categories
        tables = {}
        # Define the list of categories to filter
        categories = [
            "Table", "FigureCaption", "NarrativeText", "ListItem", "Title", "Address",
            "PageBreak", "Header", "Footer", "UncategorizedText", "Image", "Formula"
        ]

        # Traverse each category, filter out the elements of the corresponding category and store them in the dictionary
        for category in categories:
            el_list = []
            # Check whether it is in the category of "Table". You need to deal with "table.metadata.text_as_html" separately.
            tables_temp = [el for el in elements if el.category == category]
            for j in tables_temp:
                el_list.append(j.text)
            tables[category] = el_list.copy()  # Use .copy() to avoid modifying the same list object
            if category == "Table":
                # Additional storage "table.metadata.text_as_html"
                tables["Table_text_as_html"] = [el.metadata.text_as_html for el in elements if el.category == category]
    else:
        tables = "\n\n".join([str(el) for el in elements])
    print(tables)
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="epub2txt")
    parser.add_argument("--input", "-i", type=str, required=True, help="input Catalogue path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")

    # Modify-- the all parameter is of Boolean type and change the measure name to avoid conflicts with the built-in function
    parser.add_argument("--process_all", action="store_true",
                        help="process all articles if specified, otherwise split by category list, default is all articles")

    args = parser.parse_args()
    result = epub2txt(args.input, args.process_all)

    # Save the results to a file
    output_file_path = args.output  # Specify the output file path, which can be modified according to actual needs
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        if isinstance(result, dict):
            # If the result is a dictionary type, write the contents of the dictionary to the file
            json.dump(result, output_file, ensure_ascii=False, indent=4)
        else:
            # If the result is not a dictionary, write directly to the file
            output_file.write(result)
