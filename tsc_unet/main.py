import math

from matplotlib import pyplot as plt

from classifiers import unet_DistalPhalanxTW
from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets


# import os#使用cpu

# 原始
def preprocess1(x_train, length, step, pad1, pad2):
    print('训练集二维序列输入，用于生成不尺寸同的矩阵，长度，步长，pad', length, step, pad1, pad2)
    x_trainn = [[[]] for i in range(len(x_train))]
    m = length  # 20
    n = step  # 步长10
    s1 = pad1  # 需要的长度pad，32
    s2 = pad2
    for i in range(len(x_train)):
        x = math.ceil((len(x_train[i]) - n) / m) + 1
        # print(len(x_train[i]))
        # for j in range(0,x,1):#可以取几段
        count = 0  # cuowu
        for j in range(0, len(x_train[i]), n):  # 此处的终止应该为,设置起始
            # 位置
            lst = []
            # 1最后一段不足的情况
            if j + m > len(x_train[i]):
                # count = count + 1
                lst = x_train[i][j:len(x_train[i]):1]
                lst = np.pad(lst, (0, m - len(x_train[i]) + j), 'constant')
                x_trainn[i].insert(count, lst)
                # print(66666666666666666666666666666666666666666666666)
                break
            else:
                # 2 恰好能够到达最后
                # 使用切片(L-n)/m+1，取上届
                if j == 0:  # 初始

                    x_trainn[i][0] = x_train[i][0:m:1]  # 致命错误
                    count = count + 1

                elif j + m == len(x_train[i]):
                    # count = count + 1
                    lst = x_train[i][j:j + m:1]
                    x_trainn[i].insert(count, lst)
                    # print(77777777777777777777777777777777777777777777777777777777777777)
                    break
                else:  # 一般情况

                    lst = x_train[i][j:j + m:1]
                    x_trainn[i].insert(count, lst)  ##未改
                    count = count + 1
                    # print(999999999999999999999999999999999999999999999999999999999999999)
                    # print(x_trainn)
                    # return 0

        # 样本padt填充
        # pad_width=((1, 2),#向上填充1个维度，向下填充两个维度
        # (2, 1))#向左填充2个维度，向右填充一个维度
        if i == 1:
            aa = np.array(x_trainn[i])
            print('填充前shape', aa.shape)
        x_trainn[i] = np.pad(x_trainn[i], pad_width=((0, s1 - len(x_trainn[i])), (0, s2 - len(x_trainn[i][0]))),
                             mode="constant")

    # plt.plot(x_trainn[0])
    # plt.show()
    a = np.array(x_trainn)  # 每个小段里可能长度不一样，所以可能只输出前两维度
    print('形状填充后train.shape', a.shape)
    return a
    # print(x_trainn)

def preprocess2(x_train, length, step, pad1, pad2):
    print('训练集二维序列输入，用于生成不尺寸同的矩阵，长度，步长，pad', length, step, pad1, pad2)
    x_trainn = [[[]] for i in range(len(x_train))]
    m = length  # 20
    n = step  # 步长10
    s1 = pad1  # 需要的长度pad，32
    s2 = pad2
    for i in range(len(x_train)):
        x = math.ceil((len(x_train[i]) - n) / m) + 1
        # print(len(x_train[i]))
        # for j in range(0,x,1):#可以取几段
        count = 0  # cuowu
        for j in range(0, len(x_train[i]), n):  # 此处的终止应该为,设置起始
            # 位置
            lst = []
            # 1最后一段不足的情况
            if j + m > len(x_train[i]):
                # count = count + 1
                lst = x_train[i][j:len(x_train[i]):1]
                lst = np.pad(lst, (0, m - len(x_train[i]) + j), 'constant')
                x_trainn[i].insert(count, lst)
                # print(66666666666666666666666666666666666666666666666)

                if i == 0 :
                    x=range(j,j+m,1)
                    plt.plot(x,lst)

                    plt.savefig(output_directory + 'length{}step{}rank{}.jpg'.format(length, n, j))
                    plt.show()
                    print('final one 1')

                break
            else:
                # 2 恰好能够到达最后
                # 使用切片(L-n)/m+1，取上届
                if j == 0:  # 初始

                    x_trainn[i][0] = x_train[i][0:m:1]  # 致命错误

                    count = count + 1

                    if i==0:

                        x = range(j, j+m, 1)
                        plt.plot(x,x_trainn[i][0])

                        plt.savefig(output_directory + 'length{}step{}rank{}.jpg'.format(length, n,j))
                        plt.show()
                        print('j==0')

                elif j + m == len(x_train[i]):
                    # count = count + 1
                    lst = x_train[i][j:j + m:1]
                    x_trainn[i].insert(count, lst)

                    if i == 0:
                        x = range(j, j+m, 1)
                        plt.plot(x,lst)

                        plt.savefig(output_directory + 'length{}step{}rank{}.jpg'.format(length, n, j))
                        plt.show()
                        print('final one 2')
                    # print(77777777777777777777777777777777777777777777777777777777777777)
                    break
                else:  # 一般情况

                    lst = x_train[i][j:j + m:1]
                    x_trainn[i].insert(count, lst)  ##未改
                    count = count + 1

                    if i==0 and j<4*n:###
                        x = range(j, j+m, 1)
                        plt.plot(x,lst)

                        plt.savefig(output_directory + 'length{}step{}rank{}.jpg'.format(length, n,j))
                        plt.show()
                        print('general1 ')

                    elif i==0 and j>6*n and j<8*n:
                        x = range(j, j + m, 1)
                        plt.plot(x, lst)

                        plt.savefig(output_directory + 'length{}step{}rank{}.jpg'.format(length, n, j))
                        plt.show()
                        print('general2 ')

        # 样本padt填充
        # pad_width=((1, 2),#向上填充1个维度，向下填充两个维度
        # (2, 1))#向左填充2个维度，向右填充一个维度
        if i == 1:
            aa = np.array(x_trainn[i])
            print('填充前shape', aa.shape)
        # x_trainn[i] = np.pad(x_trainn[i], pad_width=((0, s1 - len(x_trainn[i])), (0, s2 - len(x_trainn[i][0]))),
        #                      mode="constant")

    plt.plot(x_trainn[0])

    plt.savefig(output_directory + 'length{}pad{}.jpg'.format(length,pad1))
    plt.show()

    print('图片')


def preprocess(x_train, length, step, pad):
    print('训练集二维序列输入，用于生成不尺寸同的矩阵，长度，步长，pad', length, step, pad)
    x_trainn = [[[]] for i in range(len(x_train))]
    m = length  # 20
    n = step  # 步长10
    s = pad  # 需要的长度pad，32
    for i in range(len(x_train)):
        x = math.ceil((len(x_train[i]) - n) / m) + 1
        # print(len(x_train[i]))
        # for j in range(0,x,1):#可以取几段
        count = 0  # cuowu
        for j in range(0, len(x_train[i]), n):  # 此处的终止应该为,设置起始
            # 位置
            lst = []
            # 1最后一段不足的情况
            if j + m > len(x_train[i]):
                # count = count + 1
                lst = x_train[i][j:len(x_train[i]):1]
                lst = np.pad(lst, (0, m - len(x_train[i]) + j), 'constant')
                x_trainn[i].insert(count, lst)
                # print(66666666666666666666666666666666666666666666666)
                break
            else:
                # 2 恰好能够到达最后
                # 使用切片(L-n)/m+1，取上届
                if j == 0:  # 初始

                    x_trainn[i][0] = x_train[i][0:m:1]  # 致命错误
                    count = count + 1
                #  print(888888888888888888888888888888888888888888888888)
                # print(x_trainn)
                # print(len(x_train[0]))
                # a=np.array(x_trainn)
                # print(a.shape)
                # return 1
                # print(x_trainn)
                # return
                elif j + m == len(x_train[i]):
                    # count = count + 1
                    lst = x_train[i][j:j + m:1]
                    x_trainn[i].insert(count, lst)
                    # print(77777777777777777777777777777777777777777777777777777777777777)
                    break
                else:  # 一般情况

                    lst = x_train[i][j:j + m:1]
                    x_trainn[i].insert(count, lst)  ##未改
                    count = count + 1
                    # print(999999999999999999999999999999999999999999999999999999999999999)
                    # print(x_trainn)
                    # return 0

        # 样本padt填充
        # pad_width=((1, 2),#向上填充1个维度，向下填充两个维度
        # (2, 1))#向左填充2个维度，向右填充一个维度
        if i == 1:
            aa = np.array(x_trainn[i])
            print('填充前shape', aa.shape)
        x_trainn[i] = np.pad(x_trainn[i], pad_width=((0, s - len(x_trainn[i])), (0, s - len(x_trainn[i][0]))),
                             mode="constant")
    a = np.array(x_trainn)  # 每个小段里可能长度不一样，所以可能只输出前两维度
    print('形状填充后train.shape', a.shape)
    return a
    # print(x_trainn)


def preprocess_test1(x_test, length, step, pad1, pad2):
    # 处理测试集
    print('测试集二维序列输入，用于生成不尺寸同的矩阵，长度，步长，pad', length, step, pad1, pad2)
    x_testn = [[[]] for i in range(len(x_test))]
    m = length  # 20
    n = step  # 步长10
    s1 = pad1  # 需要的长度pad，32
    s2 = pad2
    for i in range(len(x_test)):
        x = math.ceil((len(x_test[i]) - n) / m) + 1
        # print(len(x_test[i]))
        # for j in range(0,x,1):#可以取几段
        count = 0  # cuowu
        for j in range(0, len(x_test[i]), n):  # 此处的终止应该为,设置起始
            # 位置
            lst = []
            # 1最后一段不足的情况
            if j + m > len(x_test[i]):
                # count = count + 1
                lst = x_test[i][j:len(x_test[i]):1]
                lst = np.pad(lst, (0, m - len(x_test[i]) + j), 'constant')
                x_testn[i].insert(count, lst)

                # print(66666666666666666666666666666666666666666666666)
                break
            else:
                # 2 恰好能够到达最后
                # 使用切片(L-n)/m+1，取上届
                if j == 0:  # 初始

                    x_testn[i][0] = x_test[i][0:m:1]  # 致命错误
                    count = count + 1
                    # print(888888888888888888888888888888888888888888888888)
                    # print(x_testn)
                # print(len(x_test[0]))
                # a=np.array(x_testn)
                # print(a.shape)
                # return 1
                # print(x_testn)
                # return
                elif j + m == len(x_test[i]):
                    # count = count + 1
                    lst = x_test[i][j:j + m:1]
                    x_testn[i].insert(count, lst)
                    # print(77777777777777777777777777777777777777777777777777777777777777)
                    break
                else:  # 一般情况

                    lst = x_test[i][j:j + m:1]
                    x_testn[i].insert(count, lst)  ##未改
                    count = count + 1
                    # print(999999999999999999999999999999999999999999999999999999999999999)
                    # print(x_testn)
                    # return 0

        if i == 1:
            aa = np.array(x_testn[i])
            print('填充前shape', aa.shape)
        # print('长 高', len(x_testn[0]), len(x_testn[0][0]))
        x_testn[i] = np.pad(x_testn[i], pad_width=((0, s1 - len(x_testn[i])), (0, s2 - len(x_testn[i][0]))),
                            mode="constant")
    # print(x_testn)
    b = np.array(x_testn)
    print('填充后测试集', b.shape)
    return b


def preprocess_test(x_test, length, step, pad):
    # 处理测试集
    print('测试集二维序列输入，用于生成不尺寸同的矩阵，长度，步长，pad', length, step, pad)
    x_testn = [[[]] for i in range(len(x_test))]
    m = length  # 20
    n = step  # 步长10
    s = pad  # 需要的长度pad，32
    for i in range(len(x_test)):
        x = math.ceil((len(x_test[i]) - n) / m) + 1
        # print(len(x_test[i]))
        # for j in range(0,x,1):#可以取几段
        count = 0  # cuowu
        for j in range(0, len(x_test[i]), n):  # 此处的终止应该为,设置起始
            # 位置
            lst = []
            # 1最后一段不足的情况
            if j + m > len(x_test[i]):
                # count = count + 1
                lst = x_test[i][j:len(x_test[i]):1]
                lst = np.pad(lst, (0, m - len(x_test[i]) + j), 'constant')
                x_testn[i].insert(count, lst)

                # print(66666666666666666666666666666666666666666666666)
                break
            else:
                # 2 恰好能够到达最后
                # 使用切片(L-n)/m+1，取上届
                if j == 0:  # 初始

                    x_testn[i][0] = x_test[i][0:m:1]  # 致命错误
                    count = count + 1
                    # print(888888888888888888888888888888888888888888888888)
                    # print(x_testn)
                # print(len(x_test[0]))
                # a=np.array(x_testn)
                # print(a.shape)
                # return 1
                # print(x_testn)
                # return
                elif j + m == len(x_test[i]):
                    # count = count + 1
                    lst = x_test[i][j:j + m:1]
                    x_testn[i].insert(count, lst)
                    # print(77777777777777777777777777777777777777777777777777777777777777)
                    break
                else:  # 一般情况

                    lst = x_test[i][j:j + m:1]
                    x_testn[i].insert(count, lst)  ##未改
                    count = count + 1
                    # print(999999999999999999999999999999999999999999999999999999999999999)
                    # print(x_testn)
                    # return 0

        if i == 1:
            aa = np.array(x_testn[i])
            print('填充前shape', aa.shape)
        x_testn[i] = np.pad(x_testn[i], pad_width=((0, s - len(x_testn[i])), (0, s - len(x_testn[i][0]))),
                            mode="constant")
    # print(x_testn)
    b = np.array(x_testn)
    print('填充后测试集', b.shape)
    return b


from PIL import Image
import random


# 裁剪
def cropt(data, w, h):
    # ,w,h
    im = np.array(data) # 把矩阵当作图片
    img_size = im.size
    #print('imgsize',img_size)
    m = np.size(im,0)  # 读取图片的宽度
    n = np.size(im,1)  # 读取图片的高度
    # w = 15 # 设置你要裁剪的小图的宽度
    # h = 15  # 设置你要裁剪的小图的高度
    #for i in range(1):  # 裁剪为张随机的小图
    x = random.randint(0, m - w)  # 裁剪起点的x坐标范围
    y = random.randint(0, n - h)  # 裁剪起点的y坐标范围
    new_map = Image.fromarray(im)##############################矩阵转化为图片
    #print('new_map',new_map.size)
    #new_map.show()
    region = new_map.crop((x, y, x + w, y + h))  # 裁剪区域
    #region.show()
    #print('region',region.size)
    matrix = np.asarray(region)#####################################图片转化为矩阵
    #print(matrix)
    #print('matrix',matrix.shape)

    # region.save("你想存储的位置加名字" + str(i) + ".jpg")  # str(i)是裁剪后的编号，此处是0到99
    return matrix


def preprocess_crop(x_train, length, step, pad1, pad2):
    print('裁剪预处理，长度，步长，pad', length, step, pad1, pad2)
    x_trainn = [[[]] for i in range(len(x_train))]
    cropp=[]
    m = length  # 20
    n = step  # 步长10
    s1 = pad1  # 需要的长度pad，32
    s2 = pad2
    for i in range(len(x_train)):
        x = math.ceil((len(x_train[i]) - n) / m) + 1
        # print(len(x_train[i]))
        # for j in range(0,x,1):#可以取几段
        count = 0  # cuowu
        for j in range(0, len(x_train[i]), n):  # 此处的终止应该为,设置起始
            # 位置
            lst = []
            # 1最后一段不足的情况
            if j + m > len(x_train[i]):
                # count = count + 1
                lst = x_train[i][j:len(x_train[i]):1]
                lst = np.pad(lst, (0, m - len(x_train[i]) + j), 'constant')
                x_trainn[i].insert(count, lst)
                # print(66666666666666666666666666666666666666666666666)
                break
            else:
                # 2 恰好能够到达最后
                # 使用切片(L-n)/m+1，取上届
                if j == 0:  # 初始

                    x_trainn[i][0] = x_train[i][0:m:1]  # 致命错误
                    count = count + 1

                elif j + m == len(x_train[i]):
                    # count = count + 1
                    lst = x_train[i][j:j + m:1]
                    x_trainn[i].insert(count, lst)
                    # print(77777777777777777777777777777777777777777777777777777777777777)
                    break
                else:  # 一般情况

                    lst = x_train[i][j:j + m:1]
                    x_trainn[i].insert(count, lst)  ##未改
                    count = count + 1

        # #
        # w=10
        # h=10
        # crop = cropt(x_trainn[i], w, h)#随即裁剪
        # # 样本padt填充
        # # pad_width=((1, 2),#向上填充1个维度，向下填充两个维度
        # # (2, 1))#向左填充2个维度，向右填充一个维度
        if i == 1:
            aa = np.array(x_trainn[i])
            print('填充前shape', aa.shape)
        #cropp.append(crop)###
    a = np.array(x_trainn)
    print('未填充 a.shape', a.shape)
    return a
def crop_(data,w,h,s1,s2):
    print('裁剪开始，生成裁剪矩阵')
    a=[]
    #data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))#不需要reshape
    #print('crop reshape',data.shape)
    for i in range(len(data)):
        b=cropt(data[i],w,h)
        a.append(b)
        a[i] = np.pad(a[i], pad_width=((0, s1 - len(a[i])), (0, s2 - len(a[i][0]))),
                      mode="constant")
    a=np.array(a)
    print('最终裁剪矩阵',a.shape)
    return a

# 滑动算术平均
def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


def move_avg(a, n, mode="same"):
    c = []
    for i in range(len(a)):
        b = np_move_avg(a[i], n, mode="same")  # 这里加上list()无法运行
        c.append(b)
        # c[i] = np.pad(c[i], pad_width=((0, s1 - len(c[i])), (0, s2 - len(c[i][0]))),
        #                     mode="constant")
    c=np.array(c)
    print('滑动平均',c.shape)
    return np.array(c)


# 采样,切片
def sample(data, k):
    b = []
    for i in range(len(data)):
        a = data[i][0:len(data[i]):k]
        b.append(a)
        # print(type(b))
        # b[i] = np.pad(b[i], pad_width=((0, s1 - len(b[i])), (0, s2 - len(b[i][0]))),
        #               mode="constant")
    b=np.array(b)
    print('采样结束',b.shape)
    return b


def fit_classifier(length, step, pad1, pad2):
    x_train = datasets_dict[dataset_name][0]  # 对应字典里的顺序
    y_train = datasets_dict[dataset_name][1]  # 第一列
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]  # 第一列
    #print(x_train.shape)
    #print('y_train', y_train)

    #print(len(x_train))
    ##遍历进行数据分割，每个样本变为二维
    m = length  # 20
    n = step  # 步长10
    s1 = pad1  # 需要的长度pad，32
    s2 = pad2

    #裁剪
    data=preprocess_crop(x_train, m, n, s1, s2)
    w=1
    h=1
    crop=crop_(data, w,h,s1,s2)#裁剪矩阵
    # data1 = preprocess_crop(x_test, m, n, s1, s2)
    # w = 8
    # h = 8
    # crop_test = crop_(data1, w, h, s1, s2)  # 裁剪矩阵


    # 采样
    k = 2
    sample_train1 = sample(x_train, k)
    print('sample_train1', sample_train1.shape)
    sample_train=preprocess1(sample_train1, int(m/2), int(math.ceil(n/2)), s1, s2)
    print('采样填充',sample_train.shape)
    sample_train1 = np.pad(sample_train1, pad_width=((0,0),(0, (len(x_train[0]) - len(sample_train1[0])))),
                           mode="constant")
    print('sample_train1',sample_train1.shape)
    # print('sample_train', sample_train[1])
    # print('sample_train', sample_train.shape)

    ###滑动平均
    avg_train1 = move_avg(x_train, 3, mode="same")
    print('avg_train1',avg_train1.shape)
    #plt.plot(avg_train1[1], label="avg_train")
    #plt.plot(x_train[1], label="x_train")
    #plt.show()
    avg_train = preprocess1(avg_train1, m, n, s1, s2)
    # avg_test = move_avg(x_train, 4, mode="same")
    #print('avg_train', avg_train.shape)
    # print('avg_test', avg_test.shape)
    # print(avg_train[1])
    # print(x_train[1])

    #plt.plot(avg_train[1], label="avg_train")
    #
    #plt.show()

    plt.plot(x_train[0])
    plt.savefig(output_directory + '{}.jpg'.format(dataset_name))
    plt.show()
    #####原始多组序列
    p = 2#2
    # 测试集
    if p == 33:
        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)  # ,distal s/2,
        x_testn2 = preprocess_test1(x_test, int(m - m / 3), n, s1, s2)  # 需添加判定条件,distal s/2,int(m - m / 3), int(n - n / 3)
        x_testn3 = preprocess_test1(x_test, int(m + m / 3), n, s1, s2)  # ,distal s/2,int(n + n / 3)
        #x_testn4 = preprocess_test1(x_test, int(m - m / 3), n, s1, s2)
        # 数据增强
        # 二维序列不同的长度
        x_trainn1 = preprocess1(x_train, m, n, s1, s2)  # ,distal s/2,
        x_trainn2 = preprocess1(x_train, int(m - m / 3), n, s1, s2)  # 需添加判定条件,dia2 * s,distal s/2,
        x_trainn3 = preprocess1(x_train, int(m + m / 3), n, s1, s2)  # ,distal s/2,int(n + n / 3)

        preprocess2(x_train, m, n, s1, s2)  # ,distal s/2,
        #preprocess2(x_train, int(m - m / 3), n, s1, s2)  # 需添加判定条件,dia2 * s,distal s/2,
        #preprocess2(x_train, int(m + m / 3), n, s1, s2)  # ,distal s/2,int(n + n / 3)
        # plt.plot(x_trainn1[0])
        # plt.show()
        # plt.plot(x_trainn2[0])
        # plt.show()
        # plt.plot(x_trainn3[0])
        # plt.show()
        #x_trainn4 = preprocess1(x_train, int(m - m / 3), n, s1, s2)

        x_testn4 = x_trainn4 = np.zeros((8,8,8))
    if p == 22:
        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)  # ,distal s/2,
        x_testn2 = preprocess_test1(x_test, int(m - m / 3), n, s1, s2)  # 需添加判定条件,distal s/2,
        # x_testn3 = preprocess_test1(x_test, int(m - m / 3), n, s1, s2)  # ,distal s/2,
        # x_testn4 = preprocess_test1(x_test, int(m - m / 3), n, s1, s2)
        # 数据增强
        # 二维序列不同的长度
        x_trainn1 = preprocess1(x_train, m, n, s1, s2)  # ,distal s/2,
        x_trainn2 = preprocess1(x_train, int(m - m / 3), n, s1, s2)  # 需添加判定条件,dia2 * s,distal s/2,
        # x_trainn3 = preprocess1(x_train, int(m - m / 3), n, s1, s2)  # ,distal s/2,
        # x_trainn4 = preprocess1(x_train, int(m - m / 3), n, s1, s2)

        x_testn3 = x_testn4  = x_trainn3 = x_trainn4 = np.zeros((8,8,8))

    if p == 1:

        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)  # ,distal s/2,
        x_testn2 = preprocess_test1(x_test, int(m - m / 3), n, s1, s2)  # 需添加判定条件,distal s/2,
        x_testn3 = preprocess_test1(x_test, int(m - m / 3), n, s1, s2)  # ,distal s/2,
        x_testn4 = preprocess_test1(x_test, int(m - m / 3), n, s1, s2)
        # 数据增强
        # 二维序列不同的长度
        x_trainn1 = preprocess1(x_train, m, n, s1, s2)  # ,distal s/2,
        x_trainn2 = preprocess1(x_train, int(m - m / 3), n, s1, s2)  # 需添加判定条件,dia2 * s,distal s/2,
        x_trainn3 = preprocess1(x_train, int(m - m / 3), n, s1, s2)  # ,distal s/2,
        x_trainn4 = preprocess1(x_train, int(m - m / 3), n, s1, s2)

    elif p == 11:#一维卷积训练增强
        x_testn1 = np.zeros((8,8,8))
        x_testn2 = np.zeros((8,8,8))
        x_testn3 = np.zeros((8,8,8))
        x_trainn4 = x_testn4 = np.zeros((8,8,8))

        x_trainn1 = avg_train1
        x_trainn2 = sample_train1
        x_trainn3 = np.zeros((8,8,8))
    elif p == 111:#一维卷积训练增强
        x_testn1 = np.zeros((8,8,8))
        x_testn2 = np.zeros((8,8,8))
        x_testn3 = np.zeros((8,8,8))
        x_trainn4 = x_testn4 = np.zeros((8,8,8))

        x_trainn1 = avg_train1
        x_trainn2 = np.zeros((8,8,8))
        x_trainn3 = np.zeros((8,8,8))
    elif p == 1111:#一维卷积训练增强
        x_testn1 = np.zeros((8,8,8))
        x_testn2 = np.zeros((8,8,8))
        x_testn3 = np.zeros((8,8,8))
        x_trainn4 = x_testn4 = np.zeros((8,8,8))

        x_trainn1 = sample_train1
        x_trainn2 = np.zeros((8,8,8))
        x_trainn3 = np.zeros((8,8,8))


    elif p == 6:
        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)
        x_testn2=preprocess_test1(x_test, m, n, s1, s2)
        x_testn3=preprocess_test1(x_test, m, n, s1, s2)

        x_trainn1 = preprocess1(x_train, m, n, s1, s2)
        x_trainn2 = avg_train
        x_trainn3 = sample_train
        x_trainn4 =  x_testn4=np.zeros((8,8,8))

    elif p == 7:
        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)
        x_testn2=preprocess_test1(x_test, m, n, s1, s2)


        x_trainn1 = preprocess1(x_train, m, n, s1, s2)
        x_trainn2 = avg_train


        x_trainn4 =  x_testn4=x_trainn3 =  x_testn3=np.zeros((8,8,8))

    elif p==2:#只使用原始erwei矩阵
        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)
        x_trainn1 = preprocess1(x_train, m, n, s1, s2)

        x_testn2 = x_testn3 = x_testn4 = x_trainn2 = x_trainn3 = x_trainn4 = np.zeros((8,8,8))
    elif p==3:

        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)
        x_trainn1 = avg_train
        x_testn2 = x_testn3 = x_testn4 = x_trainn2 = x_trainn3 = x_trainn4 = np.zeros((8,8,8))
    elif p==4:
        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)
        x_trainn1 = crop
        x_testn2 = x_testn3 = x_testn4 = x_trainn2 = x_trainn3 = x_trainn4 = np.zeros((8,8,8))
    elif p==5:
        x_testn1 = preprocess_test1(x_test, m, n, s1, s2)
        x_trainn1 = sample_train
        x_testn2 = x_testn3 = x_testn4 = x_trainn2 = x_trainn3 = x_trainn4 = np.zeros((8,8,8))


    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))  # 类的数量，数据集第一列，拼接函数
    print('nub_class', nb_classes)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(
        categories='auto')  # 函数非常实用，它可以实现将分类特征的每个元素转化为一个可以用来计算的值并且，这些特征互斥，每次只有一个激活。数据会变成稀疏的。
    # categories='auto’时，编码时特征的取值取决于你输入编码数据的特征取值，两者的取值范围是一致的。

    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()  # 将标签进行编码.reshape(-1, 1)
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()  # .reshape(-1, 1)
    print('y_train', y_train.shape)
    # print(y_train)
    print('y_test', y_test.shape)
    # print(y_test)
    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)  # 行方向的最大值的位置
    print('y_true', y_true.shape)

    if len(x_train.shape) == 2:  # if univariate，，，读取矩阵的维度数据，
        #     # add a dimension to make it multivariate with one dimension ，需要增加维度
        x_train = x_train.reshape(
            (x_train.shape[0], x_train.shape[1], 1))  # ？？？？（行，列，1）,三维，样本数，列数行1列，每个样本被竖过来了，仍可看做二维。


    if len(x_trainn1.shape) == 3:
        x_trainn1 = x_trainn1.reshape((x_trainn1.shape[0], x_trainn1.shape[1], x_trainn1.shape[2], 1))
    else:
        x_trainn1 = x_trainn1.reshape(
            (x_trainn1.shape[0], x_trainn1.shape[1], 1))
        # x_trainn1=np.array(x_trainn1)
    if len(x_trainn2.shape) == 3:
        x_trainn2 = x_trainn2.reshape((x_trainn2.shape[0], x_trainn2.shape[1], x_trainn2.shape[2], 1))
    else:
        print('222222')
        x_trainn2 = x_trainn2.reshape(
            (x_trainn2.shape[0], x_trainn2.shape[1], 1))
        # x_trainn2 = np.array(x_trainn2)
    if len(x_trainn3.shape) == 3:
        x_trainn3 = x_trainn3.reshape((x_trainn3.shape[0], x_trainn3.shape[1], x_trainn3.shape[2], 1))
    if len(x_trainn4.shape) == 3:
        x_trainn4 = x_trainn4.reshape((x_trainn4.shape[0], x_trainn4.shape[1], x_trainn4.shape[2], 1))
        # x_trainn3 = np.array(x_trainn3)

    if len(x_testn1.shape) == 3:
        x_testn1 = x_testn1.reshape((x_testn1.shape[0], x_testn1.shape[1], x_testn1.shape[2], 1))
        # x_testn1 = np.array(x_testn1)
    if len(x_testn2.shape) == 3:
        x_testn2 = x_testn2.reshape((x_testn2.shape[0], x_testn2.shape[1], x_testn2.shape[2], 1))
        x_testn2 = np.array(x_testn2)
    if len(x_testn3.shape) == 3:
        x_testn3 = x_testn3.reshape((x_testn3.shape[0], x_testn3.shape[1], x_testn3.shape[2], 1))
        # x_testn3 = np.array(x_testn3)
    if len(x_testn4.shape) == 3:
        x_testn4 = x_testn4.reshape((x_testn4.shape[0], x_testn4.shape[1], x_testn4.shape[2], 1))
    print('维度升高通道：', x_trainn1.shape)


    # 增加通道变量
    input_pre1 = x_train.shape
    input_pre2, input_pre3, input_pre4, input_pre5 = x_trainn1.shape, x_trainn2.shape, x_trainn3.shape, x_trainn4.shape
    # input_pre2.append(1)
    # input_pre3.append(1)
    # input_pre4.append(1)
    print('输入的维度', input_pre1, input_pre2, input_pre3, input_pre4, input_pre5, x_testn1.shape, x_testn2.shape,
          x_testn3.shape, x_testn4.shape)
    # 只用一个输入
    # input_shape=input_pre1

    # input_shape = list(x_trainn.shape)
    # if len(input_shape) == 2:
    # print(x_trainn.shape)
    #
    # input_shape.extend([m, 1])
    # # print(input_shape)
    # # input_shape=tuple(input_shape)
    # elif len(input_shape) == 3:
    # input_shapee =list( x_trainn.shape)  # 输入的样本形状，从此处改,二三维度，， (None, 128) 表示 128 维的向量组成的变长序列。[1:]

    # print('3', input_shape)
    # print(input_shape)

    # input_pre1, input_pre2, input_pre3, input_pre4 = x_train.shape, x_trainn1.shape, x_trainn2.shape, x_trainn3.shape
    classifier = create_classifier(classifier_name, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                   nb_classes, output_directory)

    # print(x_trainn)
    # 此处目前有四个输入
    classifier.fit(x_train, x_trainn1, x_trainn2, x_trainn3, x_trainn4, y_train, x_test,x_testn1, x_testn2, x_testn3, x_testn4,
                   y_test,
                   y_true, output_directory)  # 分类器，fit


def create_classifier(classifier_name, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5, nb_classes,
                      output_directory, verbose=True):  # false

    if classifier_name == 'unet_Yoga':
        from classifiers import unet_Yoga
        return unet_Yoga.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_TwoPatterns':
        from classifiers import unet_TwoPatterns
        return unet_TwoPatterns.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_TwoLeadECG':
        from classifiers import unet_TwoLeadECG
        return unet_TwoLeadECG.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,nb_classes, verbose)

    if classifier_name == 'unet_Symbols':
        from classifiers import unet_Symbols
        return unet_Symbols.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_SyntheticControl':
        from classifiers import unet_SyntheticControl
        return unet_SyntheticControl.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_SwedishLeaf':
        from classifiers import unet_SwedishLeaf
        return unet_SwedishLeaf.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_StarLightCurves':
        from classifiers import unet_StarLightCurves
        return unet_StarLightCurves.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_ItalyPowerDemand':
        from classifiers import unet_ItalyPowerDemand
        return unet_ItalyPowerDemand.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_FiftyWords':
        from classifiers import unet_FiftyWords
        return unet_FiftyWords.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                       nb_classes, verbose)

    if classifier_name == 'unet_FacesUCR':
        from classifiers import unet_FacesUCR
        return unet_FacesUCR.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)









    # diatom数据集
    if classifier_name == 'unet_DiatomSizeReduction':
        from classifiers import unet_DiatomSizeReduction
        return unet_DiatomSizeReduction.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4, input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_Coffee':
        from classifiers import unet_Coffee
        return unet_Coffee.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                           nb_classes, verbose)
    if classifier_name == 'unet_DistalPhalanxOutlineAgeGroup':
        from classifiers import unet_DistalPhalanxOutlineAgeGroup
        return unet_DistalPhalanxOutlineAgeGroup.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                                 input_pre4, input_pre5,
                                                                 nb_classes, verbose)
    if classifier_name == 'unet1_2_2_1':
        from classifiers import unet1_2_2_1
        return unet1_2_2_1.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                           nb_classes, verbose)
    if classifier_name == 'unet1_2_2_2':
        from classifiers import unet1_2_2_2
        return unet1_2_2_2.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                           nb_classes, verbose)
    if classifier_name == 'unet_DistalPhalanxOutlineCorrect':
        from classifiers import unet_DistalPhalanxOutlineCorrect
        return unet_DistalPhalanxOutlineCorrect.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                                input_pre4, input_pre5,
                                                                nb_classes, verbose)
    if classifier_name == 'unet_DistalPhalanxTW':
        from classifiers import unet_DistalPhalanxTW
        return unet_DistalPhalanxTW.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                    input_pre5,
                                                    nb_classes, verbose)
    if classifier_name == 'unet_DistalPhalanxTW2':
        from classifiers import unet_DistalPhalanxTW2
        return unet_DistalPhalanxTW2.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                     input_pre5,
                                                     nb_classes, verbose)
    if classifier_name == 'unet_Earthquakes':
        from classifiers import unet_Earthquakes
        return unet_Earthquakes.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                input_pre5,
                                                nb_classes, verbose)
    if classifier_name == 'unet_ECG200':
        from classifiers import unet_ECG200
        return unet_ECG200.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                           nb_classes, verbose)
    if classifier_name == 'unet_ECG2002':
        from classifiers import unet_ECG2002
        return unet_ECG2002.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                            input_pre5,
                                            nb_classes, verbose)
    if classifier_name == 'unet_ECGFiveDays':
        from classifiers import unet_ECGFiveDays
        return unet_ECGFiveDays.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                input_pre5,
                                                nb_classes, verbose)
    if classifier_name == 'unet_FordA':
        from classifiers import unet_FordA
        return unet_FordA.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                          nb_classes, verbose)
    if classifier_name == 'unet_GunPoint':
        from classifiers import unet_GunPoint
        return unet_GunPoint.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                             input_pre5,
                                             nb_classes, verbose)
    if classifier_name == 'unet_Ham':
        from classifiers import unet_Ham
        return unet_Ham.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                        nb_classes, verbose)
    if classifier_name == 'unet_MiddlePhalanxOutlineAgeGroup':
        from classifiers import unet_MiddlePhalanxOutlineAgeGroup
        return unet_MiddlePhalanxOutlineAgeGroup.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                                 input_pre4, input_pre5,
                                                                 nb_classes, verbose)
    if classifier_name == 'unet_Plane':
        from classifiers import unet_Plane
        return unet_Plane.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                          nb_classes, verbose)
    if classifier_name == 'unet_ProximalPhalanxTW':
        from classifiers import unet_Plane
        return unet_Plane.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                          nb_classes, verbose)
    if classifier_name == 'unet_ShapeletSim':
        from classifiers import unet_ShapeletSim
        return unet_ShapeletSim.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                input_pre5,
                                                nb_classes, verbose)
    if classifier_name == 'unet_Trace':
        from classifiers import unet_Trace
        return unet_Trace.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                          nb_classes, verbose)
    if classifier_name == 'unet_UWaveGestureLibraryAll':
        from classifiers import unet_UWaveGestureLibraryAll
        return unet_UWaveGestureLibraryAll.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                           input_pre4, input_pre5,
                                                           nb_classes, verbose)
    if classifier_name == 'unet_UWaveGestureLibraryX':
        from classifiers import unet_UWaveGestureLibraryX
        return unet_UWaveGestureLibraryX.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                         input_pre4, input_pre5,
                                                         nb_classes, verbose)
    if classifier_name == 'unet_UWaveGestureLibraryZ':
        from classifiers import unet_UWaveGestureLibraryZ
        return unet_UWaveGestureLibraryZ.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                         input_pre4, input_pre5,
                                                         nb_classes, verbose)
    if classifier_name == 'unet_UWaveGestureLibraryY':
        from classifiers import unet_UWaveGestureLibraryY
        return unet_UWaveGestureLibraryY.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                         input_pre4, input_pre5,
                                                         nb_classes, verbose)
    if classifier_name == 'unet_Wine':
        from classifiers import unet_Wine
        return unet_Wine.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5,
                                         nb_classes, verbose)

    if classifier_name == 'unet_WordSynonyms':
        from classifiers import unet_WordSynonyms
        return unet_WordSynonyms.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_Adiac':
        from classifiers import unet_Adiac
        return unet_Adiac.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_ArrowHead':
        from classifiers import unet_ArrowHead
        return unet_ArrowHead.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_Beef':
        from classifiers import unet_Beef
        return unet_Beef.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_BeetleFly':
        from classifiers import unet_BeetleFly
        return unet_BeetleFly.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_BirdChicken':
        from classifiers import unet_BirdChicken
        return unet_BirdChicken.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_Car':
        from classifiers import unet_Car
        return unet_Car.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_CBF':
        from classifiers import unet_CBF
        return unet_CBF.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_ChlorineConcentration':
        from classifiers import unet_ChlorineConcentration
        return unet_ChlorineConcentration.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_CinCECGTorso':
        from classifiers import unet_CinCECGTorso
        return unet_CinCECGTorso.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)


    ######################
    if classifier_name == 'unet_Computers':
        from classifiers import unet_Computers
        return unet_Computers.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_CricketX':
        from classifiers import unet_CricketX
        return unet_CricketX.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_CricketY':
        from classifiers import unet_CricketY
        return unet_CricketY.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_CricketZ':
        from classifiers import unet_CricketZ
        return unet_CricketZ.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_ECG5000':
        from classifiers import unet_ECG5000
        return unet_ECG5000.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_ElectricDevices':
        from classifiers import unet_ElectricDevices
        return unet_ElectricDevices.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_FaceAll':
        from classifiers import unet_FaceAll
        return unet_FaceAll.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_FaceFour':
        from classifiers import unet_FaceFour
        return unet_FaceFour.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_FacesUCR':
        from classifiers import unet_FacesUCR
        return unet_FacesUCR.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_FiftyWords':
        from classifiers import unet_FiftyWords
        return unet_FiftyWords.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_Fish':
        from classifiers import unet_Fish
        return unet_Fish.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_FordB':
        from classifiers import unet_FordB
        return unet_FordB.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_HandOutlines':
        from classifiers import unet_HandOutlines
        return unet_HandOutlines.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_Haptics':
        from classifiers import unet_Haptics
        return unet_Haptics.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_Herring':
        from classifiers import unet_Herring
        return unet_Herring.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_InlineSkate':
        from classifiers import unet_InlineSkate
        return unet_InlineSkate.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)


###################2


    # OliveOil
    # OSULeaf
    # PhalangesOutlinesCorrect
    # Phoneme
    # Plane
    # ProximalPhalanxOutlineAgeGroup
    # ProximalPhalanxOutlineCorrect
    # ProximalPhalanxTW
    # RefrigerationDevices
    # ScreenType
    # ShapeletSim
    # ShapesAll
    # SmallKitchenAppliances
    # SonyAIBORobotSurface1
    # SonyAIBORobotSurface2
    # StarLightCurves
    # Strawberry
    # SwedishLeaf
    # Symbols
    # SyntheticControl
    # ToeSegmentation1
    # ToeSegmentation2
    # Trace
    # TwoLeadECG
    # TwoPatterns
    # UWaveGestureLibraryAll
    # UWaveGestureLibraryX
    # UWaveGestureLibraryY
    # UWaveGestureLibraryZ
    # Wafer
    # Wine
    # WordSynonyms
    # Worms
    # WormsTwoClass
    # Yoga
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)

    if classifier_name == 'unet_Wafer':
        from classifiers import unet_Wafer
        return unet_Wafer.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_ItalyPowerDemand':
        from classifiers import unet_ItalyPowerDemand
        return unet_ItalyPowerDemand.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_LargeKitchenAppliances':
        from classifiers import unet_LargeKitchenAppliances
        return unet_LargeKitchenAppliances.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_fcn':
        from classifiers import unet_fcn
        return unet_fcn.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_resnet':
        from classifiers import unet_resnet
        return unet_resnet.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_Lightning2':
        from classifiers import unet_Lightning2
        return unet_Lightning2.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_Lightning7':
        from classifiers import unet_Lightning7
        return unet_Lightning7.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_Mallat':
        from classifiers import unet_Mallat
        return unet_Mallat.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_Meat':
        from classifiers import unet_Meat
        return unet_Meat.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_MedicalImages':
        from classifiers import unet_MedicalImages
        return unet_MedicalImages.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_MiddlePhalanxOutlineCorrect':
        from classifiers import unet_MiddlePhalanxOutlineCorrect
        return unet_MiddlePhalanxOutlineCorrect.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_MiddlePhalanxTW':
        from classifiers import unet_MiddlePhalanxTW
        return unet_MiddlePhalanxTW.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_MoteStrain':
        from classifiers import unet_MoteStrain
        return unet_MoteStrain.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_NonInvasiveFetalECGThorax1':
        from classifiers import unet_NonInvasiveFetalECGThorax1
        return unet_NonInvasiveFetalECGThorax1.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_NonInvasiveFetalECGThorax2':
        from classifiers import unet_NonInvasiveFetalECGThorax2
        return unet_NonInvasiveFetalECGThorax2.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_OliveOil':
        from classifiers import unet_OliveOil
        return unet_OliveOil.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_OSULeaf':
        from classifiers import unet_OSULeaf
        return unet_OSULeaf.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)

    if classifier_name == 'unet_PhalangesOutlinesCorrect':
        from classifiers import unet_PhalangesOutlinesCorrect
        return unet_PhalangesOutlinesCorrect.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_Phoneme':
        from classifiers import unet_Phoneme
        return unet_Phoneme.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_ProximalPhalanxOutlineCorrect':
        from classifiers import unet_ProximalPhalanxOutlineCorrect
        return unet_ProximalPhalanxOutlineCorrect.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_ProximalPhalanxOutlineAgeGroup':
        from classifiers import unet_ProximalPhalanxOutlineAgeGroup
        return unet_ProximalPhalanxOutlineAgeGroup.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_RefrigerationDevices':
        from classifiers import unet_RefrigerationDevices
        return unet_RefrigerationDevices.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_ScreenType':
        from classifiers import unet_ScreenType
        return unet_ScreenType.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_ShapesAll':
        from classifiers import unet_ShapesAll
        return unet_ShapesAll.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_SmallKitchenAppliances':
        from classifiers import unet_SmallKitchenAppliances
        return unet_SmallKitchenAppliances.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_SonyAIBORobotSurface1':
        from classifiers import unet_SonyAIBORobotSurface1
        return unet_SonyAIBORobotSurface1.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_SonyAIBORobotSurface2':
        from classifiers import unet_SonyAIBORobotSurface2
        return unet_SonyAIBORobotSurface2.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)

    ###########
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3, input_pre4,
                                                 input_pre5,
                                                 nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)
    if classifier_name == 'unet_InsectWingbeatSound':
        from classifiers import unet_InsectWingbeatSound
        return unet_InsectWingbeatSound.Classifier_UNET(output_directory, input_pre1, input_pre2, input_pre3,
                                                        input_pre4,
                                                        input_pre5,
                                                        nb_classes, verbose)


    # if classifier_name == 'unet2':
    #     from classifiers import unet2
    #     return unet2.Classifier_UNET(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'unet1':
    #     from classifiers import unet1
    #     return unet1.Classifier_UNET(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'fcn':
    #     from classifiers import fcn
    #     return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)  # __init__
    # if classifier_name == 'mlp':
    #     from classifiers import mlp
    #     return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'resnet':
    #     from classifiers import resnet
    #     return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'mcnn':
    #     from classifiers import mcnn
    #     return mcnn.Classifier_MCNN(output_directory, verbose)
    # if classifier_name == 'tlenet':
    #     from classifiers import tlenet
    #     return tlenet.Classifier_TLENET(output_directory, verbose)
    # if classifier_name == 'twiesn':
    #     from classifiers import twiesn
    #     return twiesn.Classifier_TWIESN(output_directory, verbose)
    # if classifier_name == 'encoder':
    #     from classifiers import encoder
    #     return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'mcdcnn':
    #     from classifiers import mcdcnn
    #     return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'cnn':  # Time-CNN
    #     from classifiers import cnn
    #     return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    # if classifier_name == 'inception':
    #     from classifiers import inception
    #     return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

# change this directory for your machine
root_dir = 'E:/review/data'
# 'F:/桌面1/dl-4-tsc-master/data'
# 'F:\桌面1\dl-4-tsc-master\dl-4-tsc-master\data'
# '/b/home/uha/hfawaz-datas/dl-tsc-temp/'
if sys.argv[1] == 'run_all':  # 运行全部
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = read_all_datasets(root_dir, archive_name)

            for iter in range(ITERATIONS):
                print('\t\titer', iter)

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)

                tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + trr + '/'

                for dataset_name in utils.constants.dataset_names_for_archive[archive_name]:
                    print('\t\t\tdataset_name: ', dataset_name)

                    output_directory = tmp_output_directory + dataset_name + '/'

                    create_directory(output_directory)

                    fit_classifier()

                    print('\t\t\t\tDONE')

                    # the creation of this directory means
                    create_directory(output_directory + '/DONE')

elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1] == 'visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1] == 'viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1] == 'viz_cam':
    viz_cam(root_dir)
elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]  #

    length = int(sys.argv[4])
    step = int(sys.argv[5])
    pad1 = int(sys.argv[6])
    pad2 = int(sys.argv[7])

    itr = sys.argv[8]

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + \
                       dataset_name + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)

    if os.path.exists(test_dir_df_metrics):
        print('1')
        print('Already done')
    else:

        create_directory(output_directory)
        datasets_dict = read_dataset(root_dir, archive_name, dataset_name)  ##from utils

        fit_classifier(length, step, pad1, pad2)

        print('DONE')

        # the creation of this directory means
        create_directory(output_directory + '/DONE')
