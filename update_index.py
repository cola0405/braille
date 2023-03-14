import os
import re


def update_img_labels(dataset_dir):
    filenames = os.listdir(dataset_dir + '/img/')

    # 获取labels列表
    value_index: dict
    value_index = get_value_index()

    name_value: dict
    name_value = get_name_value()

    with open('labels/index.txt', 'w', encoding='utf-8') as f:
        i = 0
        for filename in filenames:
            if filename == '.DS_Store':
                continue
            # 不用改名也行的
            # 提取原图片名
            print(filename)
            # original_name = re.search("_[0-9]+.png", filename).group(0)[1:]
            value = name_value.get(filename)
            index = value_index.get(value)
            # 训练图片名 value index
            print(value)
            f.write(filename + ' ' + value + ' ' + index + '\n')
            i += 1


def update_valid_labels(dataset_dir):
    filenames = os.listdir(dataset_dir + '/valid/')

    # 获取labels列表
    value_index: dict
    value_index = get_value_index()

    name_value: dict
    name_value = get_name_value()

    with open('labels/valid_index.txt', 'w', encoding='utf-8') as f:
        i = 0
        for filename in filenames:
            if filename == '.DS_Store':
                continue
            print(filename)
            # 不用改名也行的
            # 提取原图片名
            #original_name = re.search("_[0-9]+.png", filename).group(0)[1:]
            original_name = filename
            value = name_value.get(original_name)
            index = value_index.get(value)
            # 训练图片名 value index
            f.write(filename + ' ' + value + ' ' + index + '\n')
            i += 1


def get_value_index():
    data_dir = 'labels/value_index.txt'
    with open(data_dir, encoding='utf-8') as f:
        dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
    class_name = dict(dir_labels)
    return class_name


def get_name_value():
    data_dir = 'labels/name_value.txt'
    with open(data_dir, encoding='utf-8') as f:
        dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
    class_name = dict(dir_labels)
    return class_name


dataset_dir = '/Users/cola/Downloads/braille'
img_dir = dataset_dir + '/img'
test_dir = dataset_dir + '/test'
valid_dir = dataset_dir + '/valid'

update_img_labels(dataset_dir)

update_valid_labels(dataset_dir)
