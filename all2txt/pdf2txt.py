from unstructured.partition.auto import partition

#
# #
# # elements = partition(filename="/Users/wuchengwei/Documents/privateText/人大/LLMQA/lunwen/LLMQA/2004.04906.pdf")
# # print("\n\n".join([str(el) for el in elements]))
#
#
# from unstructured.partition.pdf import partition_pdf
#
# # Returns a List[Element] present in the pages of the parsed pdf document
# # elements = partition_pdf("/Users/wuchengwei/Documents/privateText/人大/LLMQA/lunwen/LLMQA/2004.04906.pdf")
# # print(elements)
# # print("\n\n".join([str(el) for el in elements]))
#
#
# # from unstructured.documents.elements import NarrativeText
# # from unstructured.partition.text_type import sentence_count
# #
# # for element in elements[:100]:
# #     if isinstance(element, NarrativeText) and sentence_count(element.text) > 2:
# #         print(element)
# #         print("\n")


from unstructured.partition.pdf import partition_pdf

#
filename = "/Users/wuchengwei/Documents/privateText/人大/LLMQA/lunwen/LLMQA/2004.04906.pdf"
#
elements = partition_pdf(filename=filename, infer_table_structure=True, strategy="hi_res")
# print(elements)
# print("\n\n".join([str(el) for el in elements]))

tables = [el for el in elements if el.category == "Table"]
print("===tables===>" + str(tables))
for table in tables:
    print(table)
    print(table.metadata.text_as_html)
    # print(table[0].metadata.text_as_html)
tables_FigureCaption = [el for el in elements if el.category == "FigureCaption"]
tables_NarrativeText = [el for el in elements if el.category == "NarrativeText"]
tables_ListItem = [el for el in elements if el.category == "ListItem"]
tables_Title = [el for el in elements if el.category == "Title"]
tables_Address = [el for el in elements if el.category == "Address"]
tables_PageBreak = [el for el in elements if el.category == "PageBreak"]
tables_Header = [el for el in elements if el.category == "Header"]
tables_Footer = [el for el in elements if el.category == "Footer"]
tables_UncategorizedText = [el for el in elements if el.category == "UncategorizedText"]
tables_Image = [el for el in elements if el.category == "Image"]
tables_Formula = [el for el in elements if el.category == "Formula"]
print("===tables_FigureCaption===>" + str(tables_FigureCaption))
for FigureCaption in tables_FigureCaption:
    print(FigureCaption)
print("===tables_NarrativeText===>" + str(tables_NarrativeText))
for NarrativeText in tables_NarrativeText:
    print(NarrativeText)
print("===tables_ListItem===>" + str(tables_ListItem))
for ListItem in tables_ListItem:
    print(ListItem)
print("===tables_Title===>" + str(tables_Title))
for Title in tables_Title:
    print(Title)
print("===tables_Address===>" + str(tables_Address))
for Address in tables_Address:
    print(Address)
print("===tables_PageBreak===>" + str(tables_PageBreak))
for PageBreak in tables_PageBreak:
    print(PageBreak)
print("===tables_Header===>" + str(tables_Header))
for Header in tables_Header:
    print(Header)
print("===tables_Footer===>" + str(tables_Footer))
for Footer in tables_Footer:
    print(Footer)
print("===tables_UncategorizedText===>" + str(tables_UncategorizedText))
for UncategorizedText in tables_UncategorizedText:
    print(UncategorizedText)
print("===tables_Image===>" + str(tables_Image))
for Image in tables_Image:
    print(Image)
print("===tables_Formula===>" + str(tables_Formula))
for Formula in tables_Formula:
    print(Formula)
#
# print(tables[0].text)
# print(tables[0].metadata.text_as_html)

from unstructured.staging.base import elements_to_json, elements_from_json

# filename = "/Users/wuchengwei/Downloads/code/zhiyuan/wuchengwei_FlagData/FlagData/output/outputs1.json"
# elements_to_json(elements, filename=filename)
