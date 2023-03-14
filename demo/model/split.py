import cv2
import numpy as np


def split(a):   # 获取各行/列起点和终点
    # b是a的非0元素的下标 组成的数组 (np格式),同时也是高度的值
    b = np.transpose(np.nonzero(a))
    # print(b,type(b))
    # print(a,b.tolist())

    star = []
    end = []
    star.append(int(b[0]))
    for i in range(len(b)-1):
        cha_dic = int(b[i+1]) - int(b[i])	# 下一个位置跟前一个位置差距
        if cha_dic>1:						# 下一个位置跟前一个位置差距 大于1,就记录end和下一个star
            # print(cha_dic,int(b[i]),int(b[i+1]))
            end.append(int(b[i]))
            star.append(int(b[i+1]))
    end.append(int(b[len(b)-1]))
    # print(star) # [13, 50, 87, 124, 161]
    # print(end)  # [36, 73, 110, 147,184]

    return star,end

def get_vertical_shadow(img,img_bi):     # 垂直投影+分割
    # 1.垂直投影
    h,w = img_gray.shape
    shadow_v = img_bi.copy()
    a = [0 for z in range(0, w)]
    # print(a) #a = [0,0,0,0,0,0,0,0,0,0,...,0,0]初始化一个长度为w的数组，用于记录每一列的黑点个数

    # 记录每一列的波峰
    for j in range(0,w):        # 遍历一列
        for i in range(0,h):    # 遍历一行
            if shadow_v[i,j]==0: # 如果该点为黑点(默认白底黑字)
                a[j]+=1         # 该列的计数器加一计数
                shadow_v[i,j]=255 # 记录完后将其变为白色
                # print (j)
    for j in range(0,w):            # 遍历每一列
        for i in range((h-a[j]),h): # 从该列应该变黑的最顶部的点开始向最底部涂黑
            shadow_v[i,j]=0          # 涂黑
    # 2.开始分割
    # step2.1: 获取各列起点和终点
    star,end = split(a)
    # step2.2: 切割[y:y+h, x:x+w]
    for l in range(len(star)):  # 就是几列
        ys = star[l]
        ye = end[l]
        img_crop = img[0:h, ys:ye]
        # print(img_crop.shape)
        cv2.imwrite('img_4_'+str(l)+'.jpg',img_crop)

    cv2.imshow('img',img)
    cv2.imshow('img_bi',img_bi)
    cv2.imshow('shadow_v_4',shadow_v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return shadow_v


img=cv2.imread('D:/project/python/braille-demo/img/wan.png')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh,img_bi=cv2.threshold(img_gray,130,255,cv2.THRESH_BINARY)
shadow_v = get_vertical_shadow(img,img_bi)