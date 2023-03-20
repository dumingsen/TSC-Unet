# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import os#使用cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time 

from utils.utils import save_logs
from utils.utils import calculate_metrics

class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True):#verbose false
		self.output_directory = output_directory
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory+'model_init.hdf5')
		return

	def build_model(self, input_shape, nb_classes):
		##使用二维卷积
		#inp=input_shape#(样本数，长，宽，1)\
		print(input_shape)
		print(111111111111111111111111111111111111111111111111111111111111111111111)
		#print(inp)
		input_layer = keras.layers.Input(input_shape[1:])
		print(input_layer)
		conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, strides=1,padding="same")(input_layer)
		#conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)##只是1维卷积
		#卷积输出长度(卷积核的数目)，卷积核窗口长度，"same" 表示填充输入以使输出具有与原始输入相同的长度，actvition如未指定，则不使用激活函数 (即线性激活： a(x) = x)。

		conv1 = keras.layers.BatchNormalization()(conv1)
		#批标准化层
		conv1 = keras.layers.Activation(activation='relu')(conv1)
		#整流线性单元。

		#conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=1, padding="same")(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		#conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.Conv2D(filters=16, kernel_size=2,strides=1, padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)
		print('conv3',conv3.shape)
		gap_layer = keras.layers.GlobalAveragePooling2D()(conv3)#出错
		print('gap',gap_layer.shape)
		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
		print('out',output_layer.shape)
		#输出空间的维度数为类的数量，Softmax函数用于将分类结果归一化，形成一个概率分布
		model = keras.models.Model(inputs=input_layer, outputs=output_layer)#形成模型

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), metrics=['accuracy'])
		#model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
		#optimizer = 优化器，loss = 损失函数， metrics = ["准确率”])
		#多分裂交叉熵，，
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)
		#monitor：要监测的数量。factor：学习速率降低的因素。new_lr = lr * factor。patience：没有提升的epoch数，之后学习率将降低。min_lr：学习率的下限。

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true):#classifier.fit(x_train, y_train, x_test, y_test, y_true)
		#y为标签，x为数据
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 16#一次训练需要的样本数
		nb_epochs = 500#2000#

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))
		# Epoch（时期）：
		# 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一次 > epoch。（也就是说，所有训练样本在神经网络中都 进行了一次正向传播和一次反向传播
		# 再通俗一点，一个Epoch就是将所有训练样本训练一次的过程。
		# 然而，当一个Epoch的样本（也就是所有的训练样本）数量可能太过庞大（对于计算机而言），就需要把它分成多个小块，也就是就是分成多个Batch来进行训练
		# Batch（批 / 一批样本）：将整个训练样本分成若干个Batch。
		# Batch_Size（批大小）：每批样本的大小。
		# Iteration（一次迭代）：
		# 训练一个Batch就是一次Iteration（这个概念跟程序语言中的迭代器相似）。

		start_time = time.time()

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks) #model.fit()方法用于执行训练过程
		
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.hdf5')

		model = keras.models.load_model(self.output_directory+'best_model.hdf5')#模型加载

		y_pred = model.predict(x_val)#预测

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	def predict(self, x_test, y_true,x_train,y_train,y_test,return_df_metrics = True):
		model_path = self.output_directory + 'best_model.hdf5'
		model = keras.models.load_model(model_path)
		y_pred = model.predict(x_test)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			return y_pred