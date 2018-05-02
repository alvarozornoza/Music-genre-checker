
import os
import scipy.io.wavfile as wf
from scipy import signal


class Song:
    # Container for songs, it converts files to .wav and it also iterates and
    #   returns spectograms of 'interval'=size starting by 'offset' from the file
    #   located in 'path_to_file'
    
    # TODO offset

    def __init__(self,path_to_file,offset=0,interval=44100*10):
        self.path_to_file = path_to_file
        self.offset = offset 
        self.interval = interval 
        self.info = None # actually is a Wave_read Object

    def initFile(self):
        #create directory temp
        currentdir = os.curdir
        try:
            
            os.stat('temp')
            print("temp already exists.")
        except:
            print("Creating temporary directory in ", currentdir)
            os.mkdir('temp')

        #Now we write the .mp* file to .wav using linux command mpg123
        new_path_to_file = "temp"+ self.path_to_file[
                 self.path_to_file.rfind('/')
                :self.path_to_file.find('.')]+".wav"
        print("new_path_to_file ", new_path_to_file) 
        command = 'mpg123 -0w '+new_path_to_file + ' ' + self.path_to_file
        print(command)
        result = os.system(command)
        print("The RESULT is: ",result)
        if result != 0:
            raise NameError('ErrorParsingMPG123_to_system')
        self.fs , self.info =wf.read(new_path_to_file)
        self.counter = 0 + self.offset*1
    # ITERATION obj
    def __iter__(self):
        return self


    def next(self):
        return signal.spectrogram(self.readSeries(),self.fs)


    def readSeries(self):	# return frames
        if self.info is None:
            self.initFile()
        if  self.counter> len(self.info):
            raise StopIteration
        c = self.counter
        self.counter +=self.interval
        if self.counter > len(self.info):
            return self.info[c:]
        return  self.info[c:self.counter]   

