# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import math
import os  # 使用cpu

from keras import regularizers
from keras.applications.densenet import conv_block

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import tensorflow as tf
# import tensorflow._api.v2.compat.v1 as tf
#
# tf.disable_v2_behavior()
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics
from keras.models import *
from keras.layers import *

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Classifier_UNET:

    def __init__(self, output_directory, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5, nb_classes,
                 verbose=True,
                 build=True):  # verbose false
        self.output_directory = output_directory
        if build == True:
            # c = 3  # 标志
            #参数别改
            self.model = self.build_model(input_pre1, input_pre2, input_pre3, input_pre4, input_pre5, nb_classes)
            # self.model = self.ResUNet( input_pre2, input_pre3, input_pre4, input_pre5, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def conv_block(self, x, filters, kernel_size=2, padding='same', strides=1):
        'convolutional layer which always uses the batch normalization layer'
        conv = self.bn_act(x)  # bn relu
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
        return conv

    def bn_act(self, x, act=True):
        'batch normalization layer with an optinal activation layer'
        x = tf.keras.layers.BatchNormalization()(x)
        if act == True:
            x = tf.keras.layers.Activation('relu')(x)
        return x

    def stem(self, x, filters, kernel_size=2, padding='same', strides=1):
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = self.conv_block(conv, filters, kernel_size, padding, strides)  # bn relu conv2d
        shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        output = Add()([conv, shortcut])
        return output

    def residual_block(self, x, filters, kernel_size=2, padding='same', strides=1):
        res = self.conv_block(x, filters, kernel_size, padding, strides)
        res = self.conv_block(res, filters, kernel_size, padding, 1)
        shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        shortcut = self.bn_act(shortcut, act=False)
        output = Add()([shortcut, res])
        return output

    def upsample_concat_block(self, x, xskip):
        u = UpSampling2D((2, 2))(x)
        c = Concatenate()([u, xskip])
        return c

    def ResUNet(self, input_pre2, input_pre3, input_pre4, input_pre5, nb_classes):
        f = [16, 32, 64, 128, 256]
        # inputs = Input((img_h, img_w, 1))
        c = 1
        # c=3
        b = 2
        d = 16
        # 二维序列特征提取1
        input_pre2 = Input(input_pre2[1:], name='input1')
        con2 = Conv2D(d, b, padding='same')(input_pre2)  # 2d
        con2 = keras.layers.BatchNormalization()(con2)
        con2 = keras.layers.Activation('relu')(con2)
        output2 = Model(inputs=input_pre2, outputs=con2)
        print('二维序列1', con2.shape)

        # 二位特征提取2
        input_pre3 = Input(input_pre3[1:], name='input2')
        con3 = Conv2D(d, b, padding='same')(input_pre3)
        con3 = keras.layers.BatchNormalization()(con3)
        con3 = keras.layers.Activation('relu')(con3)
        # con3 = MaxPooling2D(pool_size=(2, 2))(con3)  ###############################################当维度不匹配使用
        output3 = Model(inputs=input_pre3, outputs=con3)
        print('二维序列2', con3.shape)

        # 二维特征提取3
        input_pre4 = Input(input_pre4[1:], name='input3')
        con4 = Conv2D(d, b, padding='same')(input_pre4)
        con4 = keras.layers.BatchNormalization()(con4)
        con4 = keras.layers.Activation('relu')(con4)
        output4 = Model(inputs=input_pre4, outputs=con4)
        print('二维序列3', con4.shape)

        # 二维序列提取4
        input_pre5 = Input(input_pre5[1:], name='input4')
        con5 = Conv2D(d, b, padding='same')(input_pre5)
        con5 = keras.layers.BatchNormalization()(con5)
        con5 = keras.layers.Activation('relu')(con5)
        # con5 = MaxPooling2D(pool_size=(2, 2))(con5)
        output5 = Model(inputs=input_pre5, outputs=con5)
        print('二维序列4', con5.shape)

        # 特征组合
        if c == 3:
            output_combine = concatenate([output2.output, output3.output, output4.output])
        elif c == 2:
            output_combine = concatenate([output2.output, output3.output])
        elif c == 1:
            output_combine = output2.output
        else:
            output_combine = concatenate([output2.output, output3.output, output4.output, output5.output])
        print('特征组合', output_combine.shape)
        ## Encoder
        e0 = output_combine
        e1 = self.stem(e0, f[0])
        print('e1', e1.shape)
        e2 = self.residual_block(e1, f[1], strides=2)  # 16
        print('e2', e2.shape)
        e3 = self.residual_block(e2, f[2], strides=2)
        print('e3', e3.shape)
        e4 = self.residual_block(e3, f[3], strides=2)
        print('e4', e4.shape)
        e5 = self.residual_block(e4, f[4], strides=2)
        print('e5', e5.shape)

        ## Bridge
        b0 = self.conv_block(e5, f[4], strides=1)  # 128
        print('b0', b0.shape)
        b1 = self.conv_block(b0, f[4], strides=1)
        print('b1', b1.shape)

        ## Decoder
        u1 = self.upsample_concat_block(b1, e4)  #
        d1 = self.residual_block(u1, f[4])

        u2 = self.upsample_concat_block(d1, e3)
        d2 = self.residual_block(u2, f[3])

        u3 = self.upsample_concat_block(d2, e2)
        d3 = self.residual_block(u3, f[2])

        u4 = self.upsample_concat_block(d3, e1)
        d4 = self.residual_block(u4, f[1])

        gap_layer = keras.layers.GlobalAveragePooling2D()(d4)  # 出错
        print('gap', gap_layer.shape)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax', name='output1')(gap_layer)
        print('out', output_layer.shape)

        if c == 3:
            model = Model(inputs=[output2.input, output3.input, output4.input], outputs=output_layer)
        elif c == 2:
            model = Model(inputs=[output2.input, output3.input], outputs=output_layer)
        elif c == 1:
            model = Model(inputs=[output2.input], outputs=output_layer)
        else:
            model = Model(inputs=[output2.input, output3.input, output4.input], outputs=output_layer)
        # model = Model(inputs=[x.input, y.input], outputs=z)

        # model = keras.models.Model(inputs=input_layer, outputs=output_layer)  # 形成模型

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        # model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
        # optimizer = 优化器，loss = 损失函数， metrics = ["准确率”])
        # 多分裂交叉熵，，
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, mode='auto',
                                                      min_lr=0.0001)
        # monitor：要监测的数量。factor：学习速率降低的因素。new_lr = lr * factor。patience：没有提升的epoch数，之后学习率将降低。min_lr：学习率的下限。

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def build_model(self, input_pre1, input_pre2, input_pre3, input_pre4, input_pre5, nb_classes):
        ##使用二维卷积
        c = 1
        # c = 3
        # # 1 原始序列提取特征
        # input_pre1 = Input(input_pre1[1:])
        # con1 = Conv1D(128, 8, padding='valid')(input_pre1)  # 1d
        # con1 = keras.layers.BatchNormalization()(con1)
        # con1 = keras.layers.Activation('relu')(con1)
        # con1=keras.layers.Reshape([-1,64,-1])(con1)
        # output1 = Model(inputs=input_pre1, outputs=con1)
        # print('原始序列',con1.shape)

        b = 2  # 2
        d = 32  # 16
        # 二维序列特征提取1
        input_pre2 = Input(input_pre2[1:], name='input1')
        con2 = Conv2D(d, b, padding='same')(input_pre2)  # 2d
        con2 = keras.layers.BatchNormalization()(con2)
        con2 = keras.layers.Activation('relu')(con2)
        con2 = Conv2D(d, b, padding='same')(con2)  # 2d
        con2 = keras.layers.BatchNormalization()(con2)
        con2 = keras.layers.Activation('relu')(con2)
        # con2 = MaxPooling2D(pool_size=(2, 1))(con2)
        output2 = Model(inputs=input_pre2, outputs=con2)
        print('二维序列1', con2.shape)

        # 二位特征提取2
        input_pre3 = Input(input_pre3[1:], name='input2')
        con3 = Conv2D(d, b, padding='same')(input_pre3)
        con3 = keras.layers.BatchNormalization()(con3)
        con3 = keras.layers.Activation('relu')(con3)
        con3 = Conv2D(d, b, padding='same')(con3)
        con3 = keras.layers.BatchNormalization()(con3)
        con3 = keras.layers.Activation('relu')(con3)
        ##con3 = MaxPooling2D(pool_size=(2, 2))(con3)  ###############################################当维度不匹配使用
        output3 = Model(inputs=input_pre3, outputs=con3)
        print('二维序列2', con3.shape)

        # 二维特征提取3
        input_pre4 = Input(input_pre4[1:], name='input3')
        con4 = Conv2D(d, b, padding='same')(input_pre4)
        con4 = keras.layers.BatchNormalization()(con4)
        con4 = keras.layers.Activation('relu')(con4)
        con4 = Conv2D(d, b, padding='same')(con4)
        con4 = keras.layers.BatchNormalization()(con4)
        con4 = keras.layers.Activation('relu')(con4)
        output4 = Model(inputs=input_pre4, outputs=con4)
        print('二维序列3', con4.shape)

        # 二维序列提取4
        input_pre5 = Input(input_pre5[1:], name='input4')
        con5 = Conv2D(d, b, padding='same')(input_pre5)
        con5 = keras.layers.BatchNormalization()(con5)
        con5 = keras.layers.Activation('relu')(con5)
        # con5 = MaxPooling2D(pool_size=(2, 2))(con5)
        output5 = Model(inputs=input_pre5, outputs=con5)
        print('二维序列4', con5.shape)

        # 特征组合
        if c == 3:
            output_combine = concatenate([output2.output, output3.output, output4.output])
        elif c == 2:
            output_combine = concatenate([output2.output, output3.output])
        elif c == 1:
            output_combine = output2.output
        else:
            output_combine = concatenate([output2.output, output3.output, output4.output, output5.output])
        print('特征组合', output_combine.shape)

        # inp = input_shape  # (样本数，长，宽，1)
        print(111111111111111111111111111111111111111111111111111111111111111111111)
        # print(inp)

        conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, strides=1, padding="same")(output_combine)
        # conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)##只是1维卷积
        # 卷积输出长度(卷积核的数目)，卷积核窗口长度，"same" 表示填充输入以使输出具有与原始输入相同的长度，actvition如未指定，则不使用激活函数 (即线性激活： a(x) = x)。

        conv1 = keras.layers.BatchNormalization()(conv1)
        # 批标准化层
        conv1 = keras.layers.Activation(activation='relu')(conv1)
        # 整流线性单元。

        # conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=1, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        # conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
        conv3 = keras.layers.Conv2D(filters=16, kernel_size=2, strides=1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)
        print('conv3', conv3.shape)
        gap_layer = keras.layers.GlobalAveragePooling2D()(conv3)  # 出错
        print('gap', gap_layer.shape)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax', name='output1')(gap_layer)
        print('out', output_layer.shape)
        # 输出空间的维度数为类的数量，Softmax函数用于将分类结果归一化，形成一个概率分布
        #model = keras.models.Model(inputs=input_layer, outputs=output_layer)  # 形成模型


        if c == 3:
            model = Model(inputs=[output2.input, output3.input, output4.input], outputs=output_layer)
        elif c == 2:
            model = Model(inputs=[output2.input, output3.input], outputs=output_layer)
        elif c == 1:
            model = Model(inputs=[output2.input], outputs=output_layer)
        else:
            model = Model(inputs=[output2.input, output3.input, output4.input, output5.input], outputs=output_layer)
        # model = Model(inputs=[x.input, y.input], outputs=z)

        # model = keras.models.Model(inputs=input_layer, outputs=output_layer)  # 形成模型

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        # model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
        # optimizer = 优化器，loss = 损失函数， metrics = ["准确率”])
        # 多分裂交叉熵，，
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, mode='auto',
                                                      min_lr=0.0001)
        # monitor：要监测的数量。factor：学习速率降低的因素。new_lr = lr * factor。patience：没有提升的epoch数，之后学习率将降低。min_lr：学习率的下限。

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model


    def drow(self, history, output_directory):

        epochs = range(1, len(history.history['loss']) + 1)

        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(output_directory + 'Training and Validation loss.jpg')
        plt.figure()
        epochs = range(1, len(history.history['accuracy']) + 1)
        plt.plot(epochs, history.history['accuracy'], 'bo', label='Training acc')
        plt.plot(epochs, history.history['val_accuracy'], 'b', label='validation acc')
        plt.title('Training and validation acc')
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        plt.legend()
        plt.savefig(output_directory + 'Training and validation acc.jpg')

        plt.show()

    def fit(self, x_train, x_trainn1, x_trainn2, x_trainn3, x_trainn4, y_train, x_test, x_testn1, x_testn2, x_testn3,
            x_testn4,
            y_val,
            y_true, output_directory):  # classifier.fit(x_train, y_train, x_test, y_test, y_true)
        c = 1
        # c = 3
        # y为标签，x为数据
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 16  # 一次训练需要的样本数
        nb_epochs = 250  # 2000#

        mini_batch_size = int(min(math.ceil(x_train.shape[0] / 10), batch_size))  # 大于2
        print('mini', mini_batch_size)
        # Epoch（时期）：
        # 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次 > epoch。（也就是说，所有训练样本在神经网络中都 进行了一次正向传播和一次反向传播
        # 再通俗一点，一个Epoch就是将所有训练样本训练一次的过程。
        # 然而，当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个Batch来进行训练
        # Batch（批 / 一批样本）：将整个训练样本分成若干个Batch。
        # Batch_Size（批大小）：每批样本的大小。
        # Iteration（一次迭代）：
        # 训练一个Batch就是一次Iteration（这个概念跟程序语言中的迭代器相似）。

        start_time = time.time()
        # python main.py TSC DiatomSizeReduction unet1_2 30 12 32  12
        print('123', x_trainn1.shape)  # [x_trainn1, x_trainn2, x_trainn3], y_train
        if c == 3:
            hist = self.model.fit({'input1': x_trainn1, 'input2': x_trainn2, 'input3': x_trainn3},
                                  {'output1': y_train}, batch_size=mini_batch_size,
                                  validation_data=([x_testn1, x_testn2, x_testn3], y_val),
                                  epochs=nb_epochs,
                                  verbose=self.verbose,
                                  callbacks=self.callbacks)  # model.fit()方法用于执行训练过程
        elif c == 1:
            hist = self.model.fit({'input1': x_trainn1},  # x_trainn1
                                  {'output1': y_train}, batch_size=mini_batch_size,
                                  validation_data=([x_testn1], y_val),  # x_testn1
                                  epochs=nb_epochs,
                                  verbose=self.verbose,
                                  callbacks=self.callbacks)  # model.fit()方法用于执行训练过程
        elif c == 2:
            hist = self.model.fit({'input1': x_trainn1, 'input2': x_trainn2},
                                  {'output1': y_train}, batch_size=mini_batch_size,
                                  validation_data=([x_testn1, x_testn2], y_val),
                                  epochs=nb_epochs,
                                  verbose=self.verbose,
                                  callbacks=self.callbacks)  # model.fit()方法用于执行训练过程
        else:
            hist = self.model.fit({'input1': x_trainn1, 'input2': x_trainn2, 'input3': x_trainn3, 'input4': x_trainn4},
                                  {'output1': y_train}, batch_size=mini_batch_size,
                                  validation_data=([x_testn1, x_testn2, x_testn3, x_testn4], y_val),
                                  epochs=nb_epochs,
                                  verbose=self.verbose,
                                  callbacks=self.callbacks)  # model.fit()方法用于执行训练过程
        #
        # #########画图
        # acc = hist.history['acc']  # 获取训练集准确性数据
        # val_acc = hist.history['val_acc']  # 获取验证集准确性数据
        # loss = hist.history['loss']  # 获取训练集错误值数据
        # val_loss = hist.history['val_loss']  # 获取验证集错误值数据
        # epochs = range(1, len(acc) + 1)
        # plt.plot(epochs, acc, 'bo', label='Trainning acc')  # 以epochs为横坐标，以训练集准确性为纵坐标
        # plt.plot(epochs, val_acc, 'b', label='Vaildation acc')  # 以epochs为横坐标，以验证集准确性为纵坐标
        # plt.legend()  # 绘制图例，即标明图中的线段代表何种含义
        #
        # plt.figure()  # 创建一个新的图表
        # plt.plot(epochs, loss, 'bo', label='Trainning loss')
        # plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
        # plt.legend()  ##绘制图例，即标明图中的线段代表何种含义
        #
        # plt.show()
        self.drow(hist, output_directory)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')  # 模型加载

        if c == 3:
            y_pred = model.predict([x_testn1, x_testn2, x_testn3])
        elif c == 2:
            y_pred = model.predict([x_testn1, x_testn2])
        elif c == 1:
            # y_pred = model.predict([x_testn1])
            y_pred = model.predict([x_testn1])
        else:
            y_pred = model.predict([x_testn1, x_testn2, x_testn3, x_testn4])  # 预测

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

    def fitt(self, x_train, x_trainn1, x_trainn2, x_trainn3, x_trainn4, y_train, x_test, x_testn1, x_testn2, x_testn3,
            x_testn4,
            y_val,
            y_true, output_directory):  # classifier.fit(x_train, y_train, x_test, y_test, y_true)
        c = 1
        # c = 3
        # y为标签，x为数据
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 16  # 一次训练需要的样本数
        nb_epochs = 400  # 2000#

        mini_batch_size = int(min(math.ceil(x_train.shape[0] / 10), batch_size))  # 大于2
        print('mini', mini_batch_size)
        # Epoch（时期）：
        # 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次 > epoch。（也就是说，所有训练样本在神经网络中都 进行了一次正向传播和一次反向传播
        # 再通俗一点，一个Epoch就是将所有训练样本训练一次的过程。
        # 然而，当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个Batch来进行训练
        # Batch（批 / 一批样本）：将整个训练样本分成若干个Batch。
        # Batch_Size（批大小）：每批样本的大小。
        # Iteration（一次迭代）：
        # 训练一个Batch就是一次Iteration（这个概念跟程序语言中的迭代器相似）。

        start_time = time.time()
        # python main.py TSC DiatomSizeReduction unet1_2 30 12 32  12
        print('123', x_trainn1.shape)  # [x_trainn1, x_trainn2, x_trainn3], y_train
        if c == 3:
            hist = self.model.fit({'input1': x_trainn1, 'input2': x_trainn2, 'input3': x_trainn3},
                                  {'output1': y_train}, batch_size=mini_batch_size,
                                  validation_data=([x_testn1, x_testn2, x_testn3], y_val),
                                  epochs=nb_epochs,
                                  verbose=self.verbose,
                                  callbacks=self.callbacks)  # model.fit()方法用于执行训练过程
        elif c == 1:
            hist = self.model.fit({'input1': x_train},  # x_trainn1
                                  {'output1': y_train}, batch_size=mini_batch_size,
                                  validation_data=([x_test], y_val),  # x_testn1
                                  epochs=nb_epochs,
                                  verbose=self.verbose,
                                  callbacks=self.callbacks)  # model.fit()方法用于执行训练过程
        elif c == 2:
            hist = self.model.fit({'input1': x_trainn1, 'input2': x_trainn2},
                                  {'output1': y_train}, batch_size=mini_batch_size,
                                  validation_data=([x_testn1, x_testn2], y_val),
                                  epochs=nb_epochs,
                                  verbose=self.verbose,
                                  callbacks=self.callbacks)  # model.fit()方法用于执行训练过程
        else:
            hist = self.model.fit({'input1': x_trainn1, 'input2': x_trainn2, 'input3': x_trainn3, 'input4': x_trainn4},
                                  {'output1': y_train}, batch_size=mini_batch_size,
                                  validation_data=([x_testn1, x_testn2, x_testn3, x_testn4], y_val),
                                  epochs=nb_epochs,
                                  verbose=self.verbose,
                                  callbacks=self.callbacks)  # model.fit()方法用于执行训练过程

        self.drow(hist, output_directory)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')  # 模型加载

        if c == 3:
            y_pred = model.predict([x_testn1, x_testn2, x_testn3])
        elif c == 2:
            y_pred = model.predict([x_testn1, x_testn2])
        elif c == 1:
            # y_pred = model.predict([x_testn1])
            y_pred = model.predict([x_test])
        else:
            y_pred = model.predict([x_testn1, x_testn2, x_testn3, x_testn4])  # 预测

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()

    # def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
    #     model_path = self.output_directory + 'best_model.hdf5'
    #     model = keras.models.load_model(model_path)
    #     y_pred = model.predict(x_test)
    #     if return_df_metrics:
    #         y_pred = np.argmax(y_pred, axis=1)
    #         df_metrics = calculate_metrics(y_true, y_pred, 0.0)
    #         return df_metrics
    #     else:
    #         return y_pred
    7

# distal
