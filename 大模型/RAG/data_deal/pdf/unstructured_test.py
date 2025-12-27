from unstructured.partition.pdf import partition_pdf

filename = "/home/zhangting/LLMHub/paper/gpt3.pdf"

# infer_table_structure=True automatically selects hi_res strategy
elements = partition_pdf(filename=filename, infer_table_structure=True)
tables = [el for el in elements if el.category == "Table"]

print(tables[0].text)
print('--------------------------------------------------')
print(tables[0].metadata.text_as_html)