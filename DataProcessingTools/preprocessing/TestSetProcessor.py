from bs4 import BeautifulSoup as bs

input_file_path_name = 'C:\DevWork\DevZHAW\\5.Semester\PA_BA\DataSets\Test\\newstest2014-deen-src.de.sgm'
output_file_path = 'newstest2014-deen-src.de.txt'
file = open(input_file_path_name, 'r', encoding='utf-8')
data = file.read()
soup = bs(data)

sentences = soup.find_all('seg')
print(sentences)

with open(output_file_path, 'w', encoding='utf-8') as output:
    for sentence in sentences:
        print(sentence.text, file=output)
