all2txt directory, a variety of unstructured/semi-structured files into txt, and can be a good solution to the single-column, double-column, as well as charts and graphs interspersed with the order of the text and so on leading to problems such as the text content of the problem of incoherent problems, and at the same time, after parsing the elements of the types are "Table", "FigureCaption", "NarrativeText", " ListItem", "Title", "Address", "PageBreak", "Header", "Footer", "UncategorizedText", "Image", "Formula", etc., the tool script to provide retention of the full text, as well as in accordance with the categories of parsing and preservation of the two forms, the following side of the pdf2txt as an example (epub2txt the same reason):
1, retain the full text (default)
```bash
python pdf2txt.py -i "input_path" -o "output_file"
```
2、Reserved according to different type categories
```bash
python pdf2txt.py -i "input_path" -o "output_file" --process_all
```