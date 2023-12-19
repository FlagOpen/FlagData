&emsp;&emsp;all2txt模块下，将pdf2txt、epub2txt等非结构化/半结构化的文件转成txt，并且可以很好的解决单栏、双栏，以及图表穿插中文本的顺序等导致问题文本内容不连贯的问题。

&emsp;&emsp;同时解析后的元素种类有"Table（表格）", "FigureCaption（图片标题）", "NarrativeText【正文】", "ListItem【参考文献】", "
Title【章节标题】", "Address【邮箱地址】","PageBreak", "Header【页眉】", "Footer【页脚】", "UncategorizedText【arxiv竖排编号】", "
Image(图)", "Formula（公式）" 等，工具脚本提供保留全文，以及按照类别解析保存两种形式。

下边以pdf2txt为例（epub2txt同理）：

1、保留全文（默认）
```bash
python pdf2txt.py -i "input_path" -o "output_file"
```
结果为
```markdown
Fig. 1: The overall architecture of LayoutParser...
Fig. 2: The relationship between the three types of...
Fig. 3: Layout detection and OCR results visualization...
[1] Abadi, M., Agarwal, A., Barham, P., Brevdo...
[2] Alberti, M., Pondenkandath, V., W¨ursch...
[3] Antonacopoulos, A., Bridson, D., Papadopoulos...
```

2、按不同type类别保留
```bash
python pdf2txt.py -i "input_path" -o "output_file" --process_all
```
结果为
```json lines
{
    "FigureCaption":[
        "Fig. 1: The overall architecture of LayoutParser...",
        "Fig. 2: The relationship between the three types of...",
        "Fig. 3: Layout detection and OCR results visualization..."
    ],
    "ListItem":[
        "[1] Abadi, M., Agarwal, A., Barham, P., Brevdo...",
        "[2] Alberti, M., Pondenkandath, V., W¨ursch...",
        "[3] Antonacopoulos, A., Bridson, D., Papadopoulos..."
    ]
}
```
根据不同的type类别，用户可以自动选择提取哪种类型的数据