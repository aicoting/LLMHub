import PyPDF2
filename = "/home/zhangting/LLMHub/paper/gpt3.pdf"
pdf_file = open(filename, 'rb')

reader = PyPDF2.PdfReader(pdf_file)
page_num = 8
page = reader.pages[page_num]
text = page.extract_text()

print('--------------------------------------------------')
print(text)

pdf_file.close()