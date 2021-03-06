#/usr/bin/python3
# voting system.

import numpy as np
from collections import Counter
from sliceSpectrogram import sliceSpectrogram
from PIL import Image
from imageFilesTools import getImageData
from config import sliceSize,spectrogramsPath,slicesPath
import os
CLASSES = 10
# probabilities is a matrix of Nx10, 10 number of classes.


def getDatasetFromAudioFile(filename):
    newfilename = "tmp_"+filename
    print(newfilename)
    a =createSpectrogram(filename,newfilename) # spectrogram in data/Spectrograms/a.png
    desiredSize = 128
    filenames = sliceSpectrogram(newfilename,desiredSize)
    # getting data form slices
    data = []
    for filename in filenames:
        imgData = getImageData(filename, sliceSize)
        data.append((imgData,)) # no label
        #os.remove(filename)
    return data


def sliceSpectrogram(filename, desiredSize):
# Load the full spectrogram
    img = Image.open(spectrogramsPath+filename+'.png')

    #Compute approximate number of 128x128 samples

    width, height = img.size
    nbSamples = int(width/desiredSize)
    width - desiredSize
    # Folder must be valid.
    filePaths = []
    #For each sample
    for i in range(nbSamples):
    	print ("Creating slice: ", (i+1), "/", nbSamples, "for", filename)
    	#Extract and save 128x128 sample
    	startPixel = i*desiredSize
    	imgTmp = img.crop((startPixel, 1, startPixel + desiredSize, desiredSize + 1))
        sliceName =slicesPath+filename+"_"+str(i)+".png"
        print("\n",sliceName)
        imgTmp.save(sliceName)
        filePaths.append(sliceName)
    return filePaths

def getAbsoluteMax(probabilities):
    ''' From the matrix it gets the maximum probabilities'''
    winners_perclass = np.max(probabilities,0)
    return np.argmax(winners_perclass)

def getAverageWinner(probabilities):
    return np.argmax(probabilities.mean(0))

def getFreqWinner(probabilities):
    winners = np.argmax(probabilities,1)
    return Counter(winners).most_common(1)[0][0]


def getAverage(probabilities):
    return probabilities.mean(0)
