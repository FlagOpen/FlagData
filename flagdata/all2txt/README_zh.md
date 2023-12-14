all2txt目录下，将各种非结构化/半结构化的文件转成txt，并且可以很好的解决单栏、双栏，以及图表穿插中文本的顺序等导致问题文本内容不连贯的问题，同时解析后的元素种类有"Table", "FigureCaption", "NarrativeText", "ListItem", "Title", "Address","PageBreak", "Header", "Footer", "UncategorizedText", "Image", "Formula" 等，工具脚本提供保留全文，以及按照类别解析保存两种形式，下边以pdf2txt为例（epub2txt同理）：
1、保留全文（默认）
```bash
python pdf2txt.py -i "input_path" -o "output_file"
```
2、按不同type类别保留
```bash
python pdf2txt.py -i "input_path" -o "output_file" --process_all
```