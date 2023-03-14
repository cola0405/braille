import Augmentor
import re
import os
import shutil


class DataProcessor:
    def __init__(self):
        self.dataset = None
        self.p = None

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.p = Augmentor.Pipeline(self.dataset)

    def add_data(self, dataset, num):
        self.p = Augmentor.Pipeline(dataset)
        # 2. 增强操作
        # 旋转 概率0.7，向左最大旋转角度10，向右最大旋转角度10
        self.p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)

        # 放大 概率0.3，最小为1.1倍，最大为1.6倍；1不做变换
        # self.p.zoom(probability=0.3, min_factor=1.0, max_factor=1.3)

        # resize 同一尺寸 200 x 200
        self.p.resize(probability=1, height=64, width=64)

        # 垂直方向形变
        self.p.skew_tilt(probability=0.6, magnitude=0.2)

        # 四边形形变
        #self.p.skew_corner(probability=1, magnitude=0.5)

        # 波浪扭曲
        #self.p.random_distortion(probability=1, grid_height=5, grid_width=5, magnitude=3)

        # 扭
        # p.shear(probability=0.5,max_shear_left=15,max_shear_right=15)

        # 高斯噪声
        #self.p.gaussian_distortion(0.4, grid_width = 2, grid_height = 2, magnitude = 6, corner = 'ul', method = 'in', mex=0.5, mey=0.5, sdx=0.05,
        #                    sdy=0.05)
        # 灰度图
        #self.p.greyscale(1)

        # 亮度
        self.p.random_brightness(0.6, min_factor=0.7, max_factor=1.2)

        # 饱和
        self.p.random_color(0.6, min_factor=0.5, max_factor=1.5)

        # 对比度
        self.p.random_contrast(0.6, min_factor=0.5, max_factor=1.5)

        # 3. 指定增强后图片数目总量

        self.p.sample(num)

    def clean_output_img(self, dataset):
        img_path = dataset + '/output'
        if os.path.exists(img_path):
            print('clean ' + dataset + '/output')
            shutil.rmtree(img_path)

    # 整理数据加强的output
    def update_img_labels(self, dataset_dir):
        filenames = os.listdir(dataset_dir + '/img/output/')

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
                print(filename)
                original_name = re.search("_[0-9]+.png", filename).group(0)[1:]
                value = name_value.get(original_name)
                index = value_index.get(value)
                # 训练图片名 value index
                print(value)
                f.write(filename + ' ' + value + ' ' + index + '\n')
                i += 1

    def update_test_labels(self, dataset_dir):
        filenames = os.listdir(dataset_dir + '/test/output/')

        # 获取labels列表
        value_index: dict
        value_index = self.get_value_index()

        name_value: dict
        name_value = self.get_name_value()

        with open('labels/test_index.txt', 'w', encoding='utf-8') as f:
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

    def update_valid_labels(self, dataset_dir):
        filenames = os.listdir(dataset_dir + '/valid/output/')

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
