# -*- coding: utf-8 -*-
from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image
import eyed3

from sliceSpectrogram import createSlicesFromSpectrograms
from audioFilesTools import isMono, getGenre
from config import rawDataPath
from config import spectrogramsPath
from config import pixelPerSecond

#Tweakable parameters
desiredSize = 128

#Define
currentPath = os.path.dirname(os.path.realpath(__file__)) 

#Remove logs
eyed3.log.setLevel("ERROR")

#Create spectrogram from mp3 files
def createSpectrogram(songName,newFilename):
	#Create temporary mono track if needed
	##if isMono(songName):
	command = "cp '{}' '/tmp/{}.au'".format(songName,newFilename)
	##else:
	##	command = "sox '{}' '/tmp/{}.au' remix 1,2".format(songName,newFilename)
	p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath+'/..')
	output, errors = p.communicate()
	if errors:
		print(errors)

	#Create spectrogram
	#filename.replace(".mp3","")
	command2 = "sox '/tmp/{}.au' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".format(newFilename,pixelPerSecond,spectrogramsPath+newFilename)
	p2 = Popen(command2, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath+'/..')
	output2, errors2 = p2.communicate()
	if errors2:
		print(errors2)

	#Remove tmp mono track
	os.remove("/tmp/{}.au".format(newFilename))

#Creates .png whole spectrograms from mp3 files
def createSpectrogramsFromAudio():
	genresID = dict()
	directoryElements = os.listdir(rawDataPath)
	print(directoryElements)
	genres = [genre for genre in directoryElements if os.path.isdir(rawDataPath+genre)] # We check if it is a folder
	print(genres)
	#files = [file for file in files if file.endswith(".mp3")]
	nbGenres = len(genres)
	print(nbGenres)

	#Create path if not existing
	if not os.path.exists(os.path.dirname(spectrogramsPath)):
		try:
			os.makedirs(os.path.dirname(spectrogramsPath))
		except OSError as exc: # Guard against race condition
			if exc.errno != errno.EEXIST:
				raise

	for index,genre in enumerate(genres):
		print ("Creating spectrogram for songs of genre {}.({}/{})...".format(genre,index+1,nbGenres))
		files = os.listdir(rawDataPath + '/' + genre)
		songs = [file for file in files if file.endswith(".au")]
		nbSongs = len(songs)			
		for index,filename in enumerate(songs):
			print ("Creating spectrogram for file {}/{}...".format(index+1,nbSongs))
			songName = rawDataPath + genre + '/'+ filename
			print(songName)
			print(filename)
			genresID[genre] = genresID[genre] + 1 if genre in genresID else 1
			fileID = genresID[genre]
			newFileName = genre+"_"+str(fileID)
			createSpectrogram(songName,newFileName)

	# #Rename files according to genre
	# for index,filename in enumerate(files):
	# 	print "Creating spectrogram for file {}/{}...".format(index+1,nbFiles)
	# 	fileGenre = getGenre(rawDataPath+filename)
	# 	genresID[fileGenre] = genresID[fileGenre] + 1 if fileGenre in genresID else 1
	# 	fileID = genresID[fileGenre]
	# 	newFilename = fileGenre+"_"+str(fileID)
	# 	createSpectrogram(filename,newFilename)

#Whole pipeline .mp3 -> .png slices
def createSlicesFromAudio():
	print ("Creating spectrograms...")
	createSpectrogramsFromAudio()
	print ("Spectrograms created!")

	print ("Creating slices...")
	createSlicesFromSpectrograms(desiredSize)
	print ("Slices created!")

