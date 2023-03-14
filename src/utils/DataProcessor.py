import Augmentor
import re
import os
import shutil


class DataProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.clean_output_img()
        self.p = None

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.p = Augmentor.Pipeline(self.dataset)

    def add_data(self, num):

        # 2. 增强操作
        # 旋转 概率0.7，向左最大旋转角度10，向右最大旋转角度10
        # p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)

        # 放大 概率0.3，最小为1.1倍，最大为1.6倍；1不做变换
        # p.zoom(probability=0.3, min_factor=0.7, max_factor=1.0)

        # resize 同一尺寸 200 x 200
        self.p.resize(probability=1, height=32, width=32)

        # 垂直方向形变
        self.p.skew_tilt(probability=1, magnitude=0.1)

        # 四边形形变
        #self.p.skew_corner(probability=1, magnitude=0.5)

        # 波浪扭曲
        #self.p.random_distortion(probability=1, grid_height=5, grid_width=5, magnitude=3)

        # 扭
        # p.shear(probability=0.5,max_shear_left=15,max_shear_right=15)

        # 3. 指定增强后图片数目总量
        self.p.sample(num)
        self.update_labels()


    def clean_output_img(self):
        img_path = 'img/output'
        test_path = 'test/output'
        valid_path = 'valid/output'
        if os.path.exists(img_path):
            print('clean img outpout')
            shutil.rmtree(img_path)

        if os.path.exists(test_path):
            print('clean test output')
            shutil.rmtree(test_path)

        if os.path.exists(valid_path):
            print('clean valid output')
            shutil.rmtree(valid_path)

    def update_labels(self):
        if self.dataset == 'test':
            self.update_test_labels()
            print('update test labels')
        elif self.dataset == 'valid':
            self.update_valid_labels()
            print('update valid labels')
        elif self.dataset == 'img':
            self.update_img_labels()
            print('update img labels')

    # 整理数据加强的output
    def update_img_labels(self):
        filenames = os.listdir('img/output/')

        # 获取labels列表
        value_index: dict
        value_index = self.get_value_index()

        name_value: dict
        name_value = self.get_name_value()

        with open('labels/index.txt', 'w', encoding='utf-8') as f:
            i = 0
            for filename in filenames:
                # 不用改名也行的
                # 提取原图片名
                original_name = re.search("_[0-9]+.png", filename).group(0)[1:]
                value = name_value.get(original_name)
                index = value_index.get(value)
                # 训练图片名 value index
                f.write(filename + ' ' + value + ' ' + index + '\n')
                i += 1

    def update_test_labels(self):
        filenames = os.listdir('t_test/output/')

        # 获取labels列表
        value_index: dict
        value_index = self.get_value_index()

        name_value: dict
        name_value = self.get_name_value()

        with open('t_labels/test_index.txt', 'w', encoding='utf-8') as f:
            i = 0
            for filename in filenames:
                # 不用改名也行的
                # 提取原图片名
                original_name = re.search("_[0-9]+.png", filename).group(0)[1:]
                value = name_value.get(original_name)
                index = value_index.get(value)
                # 训练图片名 value index
                f.write(filename + ' ' + value + ' ' + index + '\n')
                i += 1

    def update_valid_labels(self):
        filenames = os.listdir('valid/output/')

        # 获取labels列表
        value_index: dict
        value_index = self.get_value_index()

        name_value: dict
        name_value = self.get_name_value()

        with open('labels/valid_index.txt', 'w', encoding='utf-8') as f:
            i = 0
            for filename in filenames:
                # 不用改名也行的
                # 提取原图片名
                original_name = re.search("_[0-9]+.png", filename).group(0)[1:]
                value = name_value.get(original_name)
                index = value_index.get(value)
                # 训练图片名 value index
                f.write(filename + ' ' + value + ' ' + index + '\n')
                i += 1

    def get_value_index(self):
        data_dir = 'labels/value_index.txt'
        with open(data_dir, encoding='utf-8') as f:
            dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
        class_name = dict(dir_labels)
        return class_name

    def get_name_value(self):
        data_dir = 'labels/name_value.txt'
        with open(data_dir, encoding='utf-8') as f:
            dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
        class_name = dict(dir_labels)
        return class_name
