import os
file_path = '/Users/cola/Downloads/yisheng'
file_names = None
for path, j, file_names in os.walk(file_path):
    pass

i = 0
with open('labels.txt', 'w') as f:
    for file_name in file_names:
        #f.write(file_name + ' ' + str(i))
        f.write(file_name + ' ' + '一声')
        i += 1
        if i < len(file_names):
            f.write('\n')
