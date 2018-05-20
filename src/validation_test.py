# -*- coding: utf-8 -*-
#/usr/local/bin/python

# This file is set to test the performance of the neural network over the validation set.

## Imports
import random
import string
import os
import sys
import pickle
import numpy as np

from model import createModel
from imageFilesTools import getImageData
from config import slicesPath,datasetPath,batchSize,filesPerGenre
from config import nbEpoch,validationRatio, testRatio
from config import sliceSize

from voting_sys import getAverageWinner,getFreqWinner,getAbsoluteMax
#from songToData import createSlicesFromAudio

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test","slice","none","directo"])
# try:
# 	args = parser.parse_args()
# except Exception as e:
# 	args.mode = [None]





#Creates name of dataset from parameters
def getDatasetName(nbPerGenre, sliceSize):
    name = "{}".format(nbPerGenre)
    name += "_{}".format(sliceSize)
    return name

#Loads dataset
def loadDataset(nbPerGenre, genres, sliceSize, mode):
    #Load existing
    datasetName = getDatasetName(nbPerGenre, sliceSize)
    if mode == "validate":
        print("[+] Loading THE validation datasets... ")
        val_X = pickle.load(open("{}val_X_{}.p".format(datasetPath,datasetName), "rb" ))
        val_y = pickle.load(open("{}val_y_{}.p".format(datasetPath,datasetName), "rb" ))
        val_z = pickle.load(open("{}val_z_{}.p".format(datasetPath,datasetName), "rb" ))
        print("    Training and validation datasets loaded! ✅")
        return val_X,val_y,val_z

#Creates and save dataset from slices
def createDatasetFromSlices(nbPerGenre, genres, sliceSize, validationRatio, testRatio):
	'''     return train_X, train_y, train_z, validation_X, validation_y,
					validation_z, test_X, test_y,test_z'''
	data = []
	for genre in genres:
		print("-> Adding {}...".format(genre))
		#Get slices in genre subfolder
		filenames = os.listdir(slicesPath+genre)
		filenames = [filename for filename in filenames if filename.endswith('.png')]
		filenames = filenames[:nbPerGenre]
		#Add data (X,y)
		for filename in filenames:
			imgData = getImageData(slicesPath+genre+"/"+filename, sliceSize)
			label = [1. if genre == g else 0. for g in genres]
			s_lab = filename[:filename.rfind("_")]
			data.append((imgData,label,s_lab))

	#Extract X and y
	X,y,z = zip(*data)
	print("    Dataset created! ✅")
	return X,y,z


#Saves dataset
def saveDataset(X, y, z, nbPerGenre, genres, sliceSize):
     #Create path for dataset if not existing
    if not os.path.exists(os.path.dirname(datasetPath)):
        try:
            os.makedirs(os.path.dirname(datasetPath))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #SaveDataset
    print("[+] Saving dataset... ")
    datasetName = getDatasetName(nbPerGenre, sliceSize)
    pickle.dump(X, open("{}val_X_{}.p".format(datasetPath,datasetName), "wb" ),protocol=2)
    pickle.dump(y, open("{}val_y_{}.p".format(datasetPath,datasetName), "wb" ),protocol=2)
    pickle.dump(z, open("{}val_z_{}.p".format(datasetPath,datasetName), "wb" ),protocol=2)
    print("    Dataset saved! ✅💾")





if __name__ == '__main__':
	genres = ['country', 'disco', 'reggae', 'rock', 'pop', 'classical', 'blues', 'hiphop', 'metal', 'jazz']
	nbClasses = len(genres)
	#Load model and weights
	model = createModel(nbClasses, sliceSize)
	print("[+] Loading weightts...")
	model.load('musicDNN.tflearn')
	print("    Weights loaded! ✅")

	# load dataset
	x,y,z = loadDataset(1100, genres, sliceSize, "validate")
	# saveDataset(x,y,z,1100,genres,sliceSize)
	y = np.argmax(y,1)
	print("[+] Predicting on validation...")
	pred = []
    bulk_size=20
    for i in xrange(0,len(x),bulk_size):
        if len(pred) == 0:
            pred = list(model.predict(x[0:int(i+bulk_size)]))
        else:
            pred = pred +list(model.predict(x[int(i):int(i+bulk_size)]))
    if len(pred) == len(x):
        print("[+] Predicting lengths matched")
    else:
        print("[+] Predicting lengths do not match..")
        pred = np.concatenate(pred , model.predict(x[len(pred):]))
    if len(pred) == len(x):
        print("[+] Predicting lengths matched")
    else:
        print("[+] Predicting lengths do not match..")
    pred = np.array(pred)
    bySongProb = {}
	labels = {}
	for ind,p in enumerate(pred):
        if z[ind] not in bySongProb:
            bySongProb[z[ind]]=[]
    	bySongProb[z[ind]].append(p)
    	labels[z[ind]] = y[ind]

	freq = 0
	av = 0
	mx = 0 #max winer
	total = 0
	for song in bySongProb.keys():
		if labels[song] == getFreqWinner(bySongProb[song]):
			freq += 1
		if labels[song] == getAverage(bySongProb[song]):
			av += 1
		if labels[song] == getAbsoluteMax(bySongProb[song]):
			mx += 1
		total += 1
	print("--------------------------")
	print("| ** RESULT OVER ALL ** ")
	print("| FREQ ratio: {}".format(float(freq)/total))
	print("| AVERAGE ratio: {}".format(float(av)/total))
	print("| MAX ratio: {}".format(float(mx)/total))
	print("--------------------------")
