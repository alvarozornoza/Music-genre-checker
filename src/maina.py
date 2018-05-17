# -*- coding: utf-8 -*-
import random
import string
import os
import sys
import numpy as np

from model import createModel
from datasetTools import getDataset
from config import slicesPath
from config import batchSize
from config import filesPerGenre
from config import nbEpoch
from config import validationRatio, testRatio
from config import sliceSize

#from songToData import createSlicesFromAudio

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import argparse

import time

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Trains or tests the CNN", nargs='+', choices=["train","test","slice","testSetOfSongs"])
#parser.add_argument("nbEpoch", help="Number of epochs",type=int)
args = parser.parse_args()

print("--------------------------")
print("| ** Config ** ")
print("| Validation ratio: {}".format(validationRatio))
print("| Test ratio: {}".format(testRatio))
print("| Slices per genre: {}".format(filesPerGenre))
print("| Slice size: {}".format(sliceSize))
print("--------------------------")

if "slice" in args.mode:
	createSlicesFromAudio()
	sys.exit()

#List genres
#genres = os.listdir(slicesPath)
genres = ['country', 'disco', 'reggae', 'rock', 'pop', 'classical', 'blues', 'hiphop', 'metal', 'jazz']

#genres = [filename for filename in genres if os.path.isdir(slicesPath+filename)]
nbClasses = len(genres)

#Create model 
model = createModel(nbClasses, sliceSize)

if "train" in args.mode:

	#Create or load new dataset
	train_X, train_y, validation_X, validation_y = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, mode="train")

	#Define run id for graphs
	#run_id = "MusicGenres - "+str(batchSize)+" "+''.join(random.SystemRandom().choice(string.ascii_uppercase) for _ in range(10))
	f_scores = open("musicDNN_scores.txt", 'w')
	for epoch in range(1,nbEpoch+1):
		#t0 = time.time()
		print ("Number of epoch: " +str(epoch)+"/"+str(nbEpoch))
		#sys.stdout.flush()
		#scores =
		model.fit(train_X, train_y, batch_size=batchSize, n_epoch=1, validation_set=(validation_X, validation_y))
		#time_elapsed = time_elapsed + time.time() - t0
		#print ("Time Elapsed: " +str(time_elapsed))
		#sys.stdout.flush()
		score_train = model.evaluate(train_X, train_Y)
		print("train loss: ",score_train[0])
		print("train acc: ",score_train[1])
		score_validation = model.evaluate(validation_X, validation_Y)
		print("validation loss: ",score_validation[0])
		print("validation acc: ",score_validation[1])
		f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_validation[0])+","+str(score_validation[1]) + "\n")

		if epoch > 10:
			model_name = "musicDNN"+"_epoch_"+str(epoch)+".tflearn"
			model.save(model_name)
			#model.save(weights_path + model_name + "_epoch_" + str(epoch))
			print("Saved model to disk in: " + model_name)
	f_scores.close()
	
	#Train the model
	#print("[+] Training the model...")
	#model.fit(train_X, train_y, n_epoch=nbEpoch, batch_size=batchSize, shuffle=True, validation_set=(validation_X, validation_y), snapshot_step=100, snapshot_epoch=True,  show_metric=True, run_id=run_id)
	print("    Model trained! ✅")

	#Save trained model
	#print("[+] Saving the weights...")
	#model.save('musicDNN.tflearn')
	#print("[+] Weights saved! ✅💾")
	
if "test" in args.mode:

	#Create or load new dataset
	test_X, test_y = getDataset(filesPerGenre, genres, sliceSize, validationRatio, testRatio, mode="test")

	f_tests = open("musicDNN_testAccuracy.txt", 'w')

	#Load weights
	for epoch in range(10,21):
		print("[+] Loading weights...")
		model_name = "musicDNN"+"_epoch_"+epoch+".tflearn"
		model.load(mode_name)
		print("    Weights loaded! ✅")

		testAccuracy = model.evaluate(test_X, test_y)
		print("[+] Test accuracy: %d Loss: %d " %[testAccuracy[0],testAcuracy[1]])
		f_tests.write(str(testAccuracy[0])+","+str(testAccuracy[1])+"\n")

	f_tests.close()


	# evali= model.evaluate(test_X,test_y)[0]
	# print("Accuracy of the model is :", evali)
	# labels = model.predict_label(test_X)
	# print("The predicted labels are :",labels)
	# prediction = model.predict(test_X)
	# print("The predicted probabilities are :", prediction)

	# # Compute confusion matrix
	# cnf_matrix = confusion_matrix(test_y, [item[0] for item in labels])
	# np.set_printoptions(precision=2)

	# # Plot non-normalized confusion matrix
	# plt.figure()
	# plot_confusion_matrix(cnf_matrix, classes=class_names,
	# 					title='Confusion matrix, without normalization')

	# # Plot normalized confusion matrix
	# plt.figure()
	# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	# 					title='Normalized confusion matrix')

	# plt.show()

if "testSetOfSongs" in args.mode:
	print("Not yet")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

