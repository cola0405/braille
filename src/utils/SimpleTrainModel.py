import keras.layers
import tensorflow as tf
import re
import numpy as np
from tensorflow.python.keras import layers, models
import cv2
import keras_resnet.models


from keras.callbacks import ReduceLROnPlateau, EarlyStopping


class TrainModel:
    def __init__(self):
        self.model = models.Sequential()
        pass

    def load_dataset(self,img_dir):
        #(self.train_images, self.train_labels) = self.getDataSetByCV(img_dir + '/img/')
        (self.train_images, self.train_labels) = self.getDataSetByCV(img_dir + '/img/output/')
        self.train_images = self.train_images / 255

        #(self.valid_images, self.valid_labels) = self.getValidDataSetByCV(img_dir + '/valid/')
        (self.valid_images, self.valid_labels) = self.getValidDataSetByCV(img_dir + '/valid/output/')
        self.valid_images = self.valid_images / 255

        # (self.test_images, self.test_labels) = self.getTestDataSetByCV(img_dir + '/test/output/')
        # self.test_images = self.test_images / 255

    def build_cnn(self, labels_num):
        # 卷积层
        # input_size = 32 卷及操作输出的维度（与步长有关）
        # kernel_size = (3,3)
        # activation 激活函数relu
        # input_shape size = (32,32) channel = 3 - RGB, 1 - RGB
        self.model.add(layers.Conv2D(64, (4, 4), activation='relu', input_shape=(64, 64, 1)))

        # 池化 - 压缩并保留信息
        # pool_size = 默认是(2,2)
        # stride 步长 默认是2
        #self.model.add(layers.MaxPooling2D((2, 2)))

        self.model.add(layers.Conv2D(64, (4, 4), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(64, (4, 4), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(32, (4, 4), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Dropout(0.25))

        # 多维化一维
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(128, activation='relu'))
        # 全连接层
        self.model.add(layers.Dense(labels_num, activation='softmax'))

        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model.summary()


    def build_lenet(self, labels_num):
        self.model.add(layers.Conv2D(96, (5, 5), strides=(1, 1), input_shape=(64, 64, 1), activation='relu',
                                     kernel_initializer='uniform',padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(32, activation='relu'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.model.summary()

    # 扩大中间层
    def build_self_lenet(self, labels_num):
        self.model.add(layers.Conv2D(256, (4, 4),
                                     strides=(1, 1),
                                     input_shape=(64, 64, 1),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer='uniform'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.5))

        self.model.add(
            layers.Conv2D(128, (4, 4),
                          strides=(1, 1),
                          padding='same',
                          activation='relu',
                          kernel_initializer='uniform'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.5))

        self.model.add(
            layers.Conv2D(128, (3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu',
                          kernel_initializer='uniform'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.5))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(100, activation='relu'))
        self.model.add(layers.Dense(100, activation='relu'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.summary()

    # 中间两个dropout从0.25到0.5 - 拟合效果差
    # 回滚

    # 全部dropout调整为0.25 - 拟合效果差
    # 回滚

    # 全连接softmax前的relu调整为tanh - 拟合效果差
    # batch 从16调整到32 - 拟合效果相对好些
    # batch 从32 调整到64 - 拟合效果好，但是识别效果差
    # 回滚

    # 更换了纸张 - 优化不大
    # 拍了多张声调 - 优化大
    def build_cracker_net(self, labels_num):
        self.model.add(layers.Conv2D(64, (4, 4), strides=(1, 1), input_shape=(64, 64, 1), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(128, (4, 4), strides=(1, 1),activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Conv2D(128, (4, 4), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.summary()

    def build_alex_net(self, labels_num):
        self.model.add(
            layers.Conv2D(96, (6, 6),
                          strides=(4, 4),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))


        self.model.add(layers.Conv2D(256, (5, 5),
                                     strides=(1, 1),
                                     activation='relu',
                                     padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))


        self.model.add(layers.Conv2D(384, (3, 3),
                                     strides=(1, 1),
                                     activation='relu',
                                     padding='same'))
        self.model.add(layers.Conv2D(384, (3, 3),
                                     strides=(1, 1),
                                     activation='relu',
                                     padding='same'))
        self.model.add(layers.Conv2D(256, (3, 3),
                                     strides=(1, 1),
                                     activation='relu',
                                     padding='same'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(4096, activation='relu'))
        self.model.add(layers.Dense(4096, activation='tanh'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def build_binary_cf_net(self, labels_num):
        self.model.add(
            layers.Conv2D(64, (5, 5),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(64, (3, 3),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(64, (3, 3),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(576, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.BatchNormalization())
        # relu
        self.model.add(layers.Dense(288, activation='relu'))
        self.model.add(layers.Dense(labels_num, activation='sigmoid'))

        self.model.compile(optimizer='rmsprop',
                           loss='binary_crossentropy',
                           #loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def build_braille_multi_cf_net(self, labels_num):
        self.model.add(
            layers.Conv2D(64, (5, 5),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(64, (3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(64, (3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(576, activation='relu'))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.BatchNormalization())

        # relu
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))

        self.model.compile(optimizer='sgd',
                           #loss='categorical_crossentropy',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def build_braille_multi_cf_net_1(self, labels_num):
        self.model.add(
            layers.Conv2D(128, (3, 3),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(
            layers.Conv2D(64, (3, 3),
                          strides=(2, 2),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(32, (3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.Dropout(0.25))
        #self.model.add(layers.BatchNormalization())


        self.model.add(layers.Flatten())
        # 576
        self.model.add(layers.Dense(64, activation='relu'))
        #self.model.add(layers.Dropout(0.5))
        #self.model.add(layers.BatchNormalization())

        # 256
        #self.model.add(layers.Dense(10, activation='relu'))
        #self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(labels_num, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def build_braille_multi_cf_net_2(self, labels_num):
        self.model.add(
            layers.Conv2D(128, (4, 4),
                          strides=(2, 2),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(
            layers.Conv2D(64, (4, 4),
                          strides=(2, 2),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(32, (3, 3),
                          strides=(2, 2),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def build_braille_multi_cf_net_3(self, labels_num):
        self.model.add(
            layers.Conv2D(128, (2, 2),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))

        self.model.add(
            layers.Conv2D(64, (4, 4),
                          strides=(3, 3),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(32, (3, 3),
                          strides=(3, 3),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.Dropout(0.25))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    # dropout 到0.5 - 没用
    # 回滚回原始
    # 更换最后的激活函数到tanh - 比之前好
    # dropout 到0.5 没用
    # 增加卷积核？提取更多特征值
    # 第一个64 到128

    def build_kaggle_net2(self, labels_num):
        self.model.add(
            layers.Conv2D(128, (5, 5),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(
            layers.Conv2D(64, (3, 3),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(64, (3, 3),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.BatchNormalization())

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(576, activation='relu'))
        self.model.add(layers.Dropout(0.25))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(288, activation='tanh'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.summary()

    def build_braille_multi_cf_net_4_ing(self, labels_num):
        self.model.add(
            layers.Conv2D(128, (3, 3),
                          strides=(1, 1),
                          input_shape=(64, 64, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(2,2))
        self.model.add(layers.BatchNormalization())

        self.model.add(
            layers.Conv2D(256, (3, 3),
                          strides=(1, 1),
                          padding='same',
                          activation='relu'))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Dropout(0.5))

        self.model.add(
            layers.Conv2D(256, (2, 2),
                          strides=(1, 1),
                          padding='same',
                          activation='tanh'))
        self.model.add(layers.MaxPooling2D(2, 2))

        self.model.add(layers.Flatten())
        #self.model.add(layers.BatchNormalization())
        #self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(labels_num, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

    def build_kersa_resnet(self, labels_num):
        input = keras.layers.Input((64, 64, 1))

        self.model = keras_resnet.models.ResNet50(inputs=input, classes=labels_num)
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    def build_kersa_vgg16(self, labels_num):
        from keras.applications.vgg16 import VGG16
        input_shape = keras.layers.Input((64, 64, 1))

        self.model: keras.Model

        # have to be 3
        self.model = VGG16(include_top=False, input_shape=(64, 64, 3))
        print(self.model.layers[-1].output)

        from keras import layers
        flat1 = layers.Flatten()(self.model.layers[-1].output)
        class1 = layers.Dense(1024, activation='relu')(flat1)
        output = layers.Dense(labels_num, activation='softmax')(class1)
        self.model = keras.Model(inputs=self.model.inputs, outputs=output)
        self.model.summary()

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


    def keras_app_test(self, labels_num):
        from keras.applications.resnet import ResNet50
        from keras.preprocessing import image
        from keras.applications.resnet import preprocess_input, decode_predictions
        import numpy as np

        model = ResNet50(weights='imagenet')

        img_path = 'img/img_1.png'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        preds = model.predict(x)
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        print('Predicted:', decode_predictions(preds, top=3)[0])

    from keras.applications.resnet import ResNet50

    def train_model(self, model_name, labels_num):
        #self.build_cnn(labels_num)
        #self.build_lenet(labels_num)
        #self.build_self_lenet(labels_num)
        #self.build_cracker_net(labels_num)
        #self.build_alex_net(labels_num)
        #self.build_kaggle_net2(labels_num)
        #self.build_binary_cf_net(labels_num)
        #self.build_braille_multi_cf_net_1(labels_num)
        #self.build_braille_multi_cf_net_2(labels_num)
        #self.build_braille_multi_cf_net_3(labels_num)
        #self.build_braille_multi_cf_net_4_ing(labels_num)
        #self.build_kersa_resnet(labels_num)
        self.build_kersa_vgg16(labels_num)
        #self.keras_app_test(labels_num)

        early_stop = EarlyStopping(patience=5, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
        self.model.fit(self.train_images,
                       self.train_labels,
                       epochs=600,
                       verbose=1,
                       batch_size=128,
                       callbacks=[reduce_lr,early_stop],
                       shuffle=True,
                       validation_data=(self.valid_images, self.valid_labels))

        # 保存模型
        checkpoint_path = 'models/' + model_name
        #checkpoint_path = 'models/' + 'cp.ckpt'
        self.model.save(checkpoint_path)

        # 评估模型
        #self.model.evaluate(self.test_images, self.test_labels, verbose=2)

    def getDataSetByCV(self, img_dir):
        filenames = []
        # 获取训练集对应关系
        with open('labels/index.txt', encoding='utf-8') as f:
            dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
            # 一定要用strip,因为原txt文件每行后面会带‘\n‘字符；

        images = []
        labels = np.array([])

        # 整理labels
        for l in dir_labels:
            filenames.append(l[0])
            label = [int(l[2])]
            labels = np.append(labels, label)

        # 整理images
        for filename in filenames:
            img = cv2.imread(img_dir + filename)
            #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = gray_img
            img = cv2.resize(img, (64, 64))
            # (64,64,1,3)
            #img = np.expand_dims(img, axis=2)
            images.append(img)

        images = np.array(images, dtype=object).astype('float32')

        return images, labels

    def getValidDataSetByCV(self, img_dir):
        filenames = []
        # 获取训练集对应关系
        with open('labels/valid_index.txt', encoding='utf-8') as f:
            dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
            # 一定要用strip,因为原txt文件每行后面会带‘\n‘字符；

        images = []
        labels = np.array([])

        # 整理labels
        for l in dir_labels:
            filenames.append(l[0])
            label = [int(l[2])]
            labels = np.append(labels, label)

        # 整理images
        for filename in filenames:
            img = cv2.imread(img_dir + filename)
            #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = gray_img
            img = cv2.resize(img, (64, 64))
            #img = np.expand_dims(img, axis=2)
            images.append(img)

        images = np.array(images, dtype=object).astype('float32')
        print(labels)
        return images, labels

    def getTestDataSetByCV(self, img_dir):
        filenames = []
        # 获取训练集对应关系
        with open('labels/test_index.txt', encoding='utf-8') as f:
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
        for filename in filenames:
            img = cv2.imread(img_dir + filename)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = gray_img
            img = np.expand_dims(img, axis=2)
            images.append(img)

        images = np.array(images, dtype=object).astype('float32')
        return images, labels
