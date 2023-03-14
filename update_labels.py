import re

indexs = []
values = []
names = []

with open('labels/name_value.txt', encoding='utf-8') as f:
    i = 0

    for line in f.readlines():
        filename, value = re.split(' ', line.strip())
        if value in values:
            continue
        values.append(value)
        indexs.append(str(i))
        i += 1


with open('labels/index_value.txt', 'w', encoding='utf-8')as iv:
    for i in range(len(indexs)):
        iv.write(indexs[i] + ' ' + values[i] + '\n')

with open('labels/value_index.txt', 'w', encoding='utf-8')as iv:
    for i in range(len(indexs)):
        iv.write(values[i] + ' ' + indexs[i] + '\n')
