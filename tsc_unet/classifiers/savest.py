#train
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
# print(x_trainn)

#test
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
print('b.shape', b.shape)