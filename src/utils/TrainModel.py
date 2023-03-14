import tensorflow as tf
import re
import numpy as np
from tensorflow.python.keras import layers, models
import cv2


class TrainModel:
    def __init__(self):
        (self.train_images, self.train_labels) = self.getDataSetByCV()
        (self.valid_images, self.valid_labels) = self.getValidDataSetByCV()
        (self.test_images, self.test_labels) = self.getTestDataSetByCV()
        self.model = models.Sequential()
        pass

    def build_cnn(self):
        # 卷积层
        # input_size = 32 卷及操作输出的维度（与步长有关）
        # kernel_size = (3,3)
        # activation 激活函数relu
        # input_shape size = (32,32) channel = 3 - RGB, 1 - RGB
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))

        # 池化 - 压缩并保留信息
        # pool_size = 默认是(2,2)
        # stride 步长 默认是2
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # 多维化一维
        self.model.add(layers.Flatten())

        # 全连接层
        # 维度对应与上面卷积操作输出的维度64应该一致
        self.model.add(layers.Dense(64, activation='relu'))

        # 不要求比上一层小
        # 维度对应637个labels
        self.model.add(layers.Dense(429, activation='softmax'))

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model.summary()

    def train_model(self):
        self.build_cnn()
        self.model.fit(self.train_images,
                       self.train_labels,
                       epochs=9,
                       verbose=1,
                       validation_data=(self.valid_images, self.valid_labels))

        # 保存模型
        checkpoint_path = 'models/m2.h5'
        self.model.save(checkpoint_path)

        # 评估模型
        self.model.evaluate(self.test_images, self.test_labels, verbose=2)

    def getDataSetByCV(self):
        filenames = []
        # 获取训练集对应关系
        with open('./labels/index.txt', encoding='utf-8') as f:
            dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
            # 一定要用strip,因为原txt文件每行后面会带‘\n‘字符；

        images = []
        labels = np.array([])

        # 整理labels
        for l in dir_labels:
            filenames.append(l[0])
            label = [int(l[2])]
            labels = np.append(labels, label)

        # 训练集标准规格
        re_size = (32, 32)

        # 整理images
        for img_dir in filenames:
            img = cv2.imread('./img/output/' + img_dir)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            r, threshold_img = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY_INV)
            img = cv2.resize(threshold_img, re_size, interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=2)
            images.append(img)

        images = np.array(images, dtype=object).astype('float32')
        return images, labels

    def getValidDataSetByCV(self):
        filenames = []
        # 获取训练集对应关系
        with open('./labels/valid_index.txt', encoding='utf-8') as f:
            dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
            # 一定要用strip,因为原txt文件每行后面会带‘\n‘字符；

        images = []
        labels = np.array([])

        # 整理labels
        for l in dir_labels:
            filenames.append(l[0])
            label = [int(l[2])]
            labels = np.append(labels, label)

        # 训练集标准规格
        re_size = (32, 32)

        # 整理images
        for img_dir in filenames:
            img = cv2.imread('./valid/output/' + img_dir)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            r, threshold_img = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY_INV)
            img = cv2.resize(threshold_img, re_size, interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=2)
            images.append(img)

        images = np.array(images, dtype=object).astype('float32')
        return images, labels

    def getTestDataSetByCV(self):
        filenames = []
        # 获取训练集对应关系
        with open('./labels/test_index.txt', encoding='utf-8') as f:
            dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
            # 一定要用strip,因为原txt文件每行后面会带‘\n‘字符；

        images = []
        labels = np.array([])

        # 整理labels
        for l in dir_labels:
            filenames.append(l[0])
            label = [int(l[2])]
            labels = np.append(labels, label)

        # 训练集标准规格
        re_size = (32, 32)

        # 整理images
        for img_dir in filenames:
            img = cv2.imread('./test/output/' + img_dir)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            r, threshold_img = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY_INV)
            img = cv2.resize(threshold_img, re_size, interpolation=cv2.INTER_AREA)
            img = np.expand_dims(img, axis=2)
            images.append(img)

        images = np.array(images, dtype=object).astype('float32')
        return images, labels
