Under the all2txt module, the unstructured / semi-structured files such as pdf2txt and epub2txt can be converted into txt, and it can well solve the problem of incoherent text content caused by single column, double column, and the order of Chinese text interspersed with charts.

At the same time, the types of elements after parsing are "Table", "FigureCaption", "NarrativeText", "ListItem", "
Title [Chapter Title]", "Address [E-mail]","PageBreak", "Header [Header]", "Footer [Footer]", "UncategorizedText [arxiv vertical number]", "
Image, Formula, etc. Tool scripts provide two forms: keeping full text and saving by category resolution.

Take pdf2txt as an example (the same goes for epub2txt):

1. retain the full text (default)
```bash
python pdf2txt.py -i "input_path" -o "output_file"
```
The result is
```markdown
Fig. 1: The overall architecture of LayoutParser...
Fig. 2: The relationship between the three types of...
Fig. 3: Layout detection and OCR results visualization...
[1] Abadi, M., Agarwal, A., Barham, P., Brevdo...
[2] Alberti, M., Pondenkandath, V., W¨ursch...
[3] Antonacopoulos, A., Bridson, D., Papadopoulos...
```

2. Reserved according to different type categories
```bash
python pdf2txt.py -i "input_path" -o "output_file" --process_all
```
The result is
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
According to different type categories, users can automatically choose which type of data to extract.