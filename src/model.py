# -*- coding: utf-8 -*-

import numpy as np
import os

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from config import tmpPath

def createModel(nbClasses,imageSize):
	#Create path tmp if not existing
	if not os.path.exists(os.path.dirname(tmpPath)):
		try:
			os.makedirs(os.path.dirname(tmpPath))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	print("[+] Creating model...")
	convnet = input_data(shape=[None, imageSize, imageSize, 1], name='input')

	convnet = conv_2d(convnet, 64, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 128, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 256, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = conv_2d(convnet, 512, 2, activation='elu', weights_init="Xavier")
	convnet = max_pool_2d(convnet, 2)

	convnet = fully_connected(convnet, 1024, activation='elu')
	convnet = dropout(convnet, 0.5)

	convnet = fully_connected(convnet, nbClasses, activation='softmax')
	convnet = regression(convnet, optimizer='rmsprop', loss='categorical_crossentropy')

	model = tflearn.DNN(convnet,checkpoint_path=tmpPath+'tflearn_logs/',max_checkpoints=1, tensorboard_verbose=0)
	print("    Model created! ✅")
	return model