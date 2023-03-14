import cv2

from tensorflow.python.keras import models
import numpy as np
import re


# 统计横向和竖向的白色像素点
line_white_pixel_count = []
column_white_pixel_count = []


class BrailleProcessor:
    def __init__(self, model_name, labels_name):
        self.model = self.load_model(model_name)
        self.end_flag = [50, 51, 52, 53]
        self.class_name = self.get_class_names(labels_name)
        pass

    def load_model(self, model_name):
        checkpoint_path = './model/' + model_name
        model: models.Sequential
        model = models.load_model(checkpoint_path)
        return model

    @staticmethod
    def get_segments(img_path, show):
        global line_white_pixel_count
        global column_white_pixel_count
        line_white_pixel_count = []
        column_white_pixel_count = []
        img = cv2.imread(img_path)
        origin_img = img.copy()
        if show:
            cv2.imshow('img', img)

        # 输出图像尺寸和通道信息
        sp = img.shape
        img_height = sp[0]
        img_width = sp[1]
        channel = sp[2]
        print(' width: %d \n height: %d \n number: %d' % (img_width, img_height, channel))

        # 阈值化
        threshold_img = get_threshold_img(img, show)
        # cv2.imshow('threshold', threshold_img)

        # 胀化处理
        dilate_img = get_dilate_img(img_height, threshold_img, show)
        # cv2.imshow('dilate', dilate_img)

        # 求平均行高
        inline_avg_height = get_inline_avg_height(dilate_img, img_width, img_height)
        # print('平均行高: ', inline_avg_height)

        # 获取各竖向切线
        rect_y = get_rect_y(img_height, inline_avg_height)

        # 横向切，并得到矩形区域切片
        rects = get_rects(rect_y, dilate_img, img_height, img_width, inline_avg_height)

        # 绘制矩形框线
        draw_rect(img, rects, show)
        return origin_img, rects


    @staticmethod
    def get_class_names(labels_name):
        data_dir = './model/' + labels_name
        with open(data_dir, encoding='utf-8') as f:
            dir_labels = [re.split(' ', line.strip()) for line in f.readlines()]
        class_name = dict(dir_labels)
        return class_name

    @staticmethod
    def get_specified_area_in_img(img, area):
        start_y = area[0]
        end_y = area[1]
        start_x = area[2]
        end_x = area[3]
        re_size = (128, 128)
        target_img = img[start_y:end_y, start_x:end_x]
        # if index == -1:
        #     cv2.imshow('target', t_img)
        # else:
        #     cv2.imshow('slice' + str(index), t_img)
        target_img = cv2.resize(target_img, re_size, interpolation=cv2.INTER_AREA)
        target_img = target_img / 255

        # target_img = np.expand_dims(target_img, axis=2).astype('float32')
        return target_img

    def predict(self, img):
        images = [img]
        images = np.array(images)

        # 模型predict
        res: np.ndarray
        #res = self.model.predict(images)[0]
        res = self.model.predict(images)

        # 输出最可能的结果 - 训练集内的可能性可以到0.9322300
        res = res[0]
        length = len(res)
        max_res = res.max()
        for i in range(length):
            if res[i] == max_res and res[i] > 0.3:
                #print('===========')
                #print(max_res)
                key = i
                return key
        # 无法识别
        NULL = 54
        return NULL

    def get_label_by_index(self, key):
        return self.class_name.get(str(key))


# 胀化处理
def get_dilate_img(img_height, threshold_img, show):
    # 宽度除以10大概就是胀化的数量级
    if img_height > 250:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    else:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate_img = cv2.dilate(threshold_img, dilate_kernel)
    if show:
        cv2.imshow("dilate_img", dilate_img)
    return dilate_img


# 平均行高
def get_inline_avg_height(dilate_img, img_width, img_height):
    inline_heights = []
    # 统计各行白色像素点个数
    for i in range(img_height):
        count = 0
        for j in range(img_width):
            if dilate_img[i, j] == 255:
                count += 1
        line_white_pixel_count.append(count)

    i = 0
    # 统计各行文本高度
    while i < img_height:
        inline_height = 0
        while i < img_height and line_white_pixel_count[i] != 0:
            inline_height += 1
            i += 1
        if inline_height != 0:
            inline_heights.append(inline_height)
        i += 1

    print(inline_heights)
    if len(inline_heights) != 0:
        inline_avg_height = sum(inline_heights) // len(inline_heights)
    else:
        inline_avg_height = 0
    return inline_avg_height


# 阈值化
def get_threshold_img(img, show):
    # RBG转灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('最亮点：')
    highest = max(gray_img.ravel())
    print(highest)
    print('平均灰度值：')
    average_gray = sum(gray_img.ravel()) // len(gray_img.ravel())
    print(average_gray)

    # 阈值化
    t = (highest + average_gray) / 2
    retval, threshold_img = cv2.threshold(gray_img, t, 255, cv2.THRESH_TOZERO)
    retval, threshold_img = cv2.threshold(threshold_img, t, 255, cv2.THRESH_BINARY)
    if show:
        cv2.imshow("threshold", threshold_img)
    return threshold_img


# 竖向切
def get_rect_y(img_height, inline_avg_height):
    rect_y = []
    i = 0
    line_start_y = 0
    line_start_flag = 0
    while i < img_height:
        # 遇到有字区域
        if line_start_flag == 0 and line_white_pixel_count[i] > 3:
            line_start_flag = 1
            line_start_y = i

        # 空白区域
        # 超出合理范围的舍弃掉
        # gray_value_x[i]<3 为背景，该行没有字
        elif line_start_flag == 1 and line_white_pixel_count[i] < 3 and i - line_start_y > inline_avg_height * 0.3:

            # 至少一行字的高度
            # 如果区域不满足条件 - 竖线高度不足5，不是一行字，重新画竖线
            if i - line_start_y > inline_avg_height * 0.9:
                line_start_flag = 0

                # rect是list，表示一区间
                # 竖向划线
                rect = [line_start_y, i + 1]

                # len(text_rect_x)为总共有几行文字
                rect_y.append(rect)
        i += 1
    return rect_y


# 切片矩形区域
def get_rects(rect_y, dilate_img, img_height, img_width, inline_avg_height):
    rects = []
    # 在竖直切线的基础上水平切
    for rect in rect_y:
        # 矩形的宽
        # 矩形的长 - 0:sp[1] 为一行从左到右
        # 横向裁出一段
        cropImg = dilate_img[rect[0]:rect[1], 0:img_width]

        # 一行文字的宽高
        sp_y = cropImg.shape

        # 统计各列有字区域的数量
        for i in range(sp_y[1]):
            white_value = 0
            for j in range(img_height):

                # 胀化后的图像，统计文字区域
                if dilate_img[j, i] == 255:
                    white_value += 1
            column_white_pixel_count.append(white_value)

        start_flag = 0
        start_x = 0
        i = 0

        # 横向切
        while i < img_width:
            if start_flag == 0 and column_white_pixel_count[i] > 3:
                start_flag = 1
                start_x = i
            elif start_flag == 1 and column_white_pixel_count[i] < 3 and i - start_x > inline_avg_height * 0.25:
                if i - start_x > inline_avg_height * 0.5:
                    start_flag = 0
                    rects.append([rect[0], rect[1], start_x - 1, i + 1])
            i += 1
    return rects


# 绘制矩形
def draw_rect(img, rects, show):
    if len(rects):
        # 在原图像中画出矩形框
        # rectangle 的参数
        # img,startPoint,EndPoint,color,thickness
        # startPoint是左上角，endPoint是右下角
        # startPoint(100,50)，表示往右x=100,往下y=50
        for rect in rects:
            rectangle_img = cv2.rectangle(img, (rect[2], rect[0]), (rect[3], rect[1]),
                                          (255, 0, 0),thickness=1)
        if show :
            cv2.imshow("Rectangle Image", rectangle_img)
