import os
import scipy.io.wavfile as wf


class Song:
    # Container for songs, it converts files to .wav and it also iterates and
    #   returns spectograms of 'interval'=size starting by 'offset' from the file
    #   located in 'path_to_file'
    
    # TODO offset

    def __init__(self,path_to_file,offset=0,interval=1024):
        self.path_to_file = path_to_file
        self.offset = offset 
        self.interval = interval 
        self.info = None # actually is a Wave_read Object

    def initFile(self):
        #create directory temp
        currentdir = os.curdir
        try:
            
            os.stat(directory)
            print("temp already exists.")
        except:
            print("Creating temporary directory in ", currentdir)
            os.mkdir('temp')

        #Now we write the .mp* file to .wav using linux command mpg123
        new_path_to_file = "temp"+ self.path_to_file[
                 self.path_to_file.rfind('/')
                :self.path_to_file.find('.')]+".wav"
        print("new_path_to_file ", new_path_to_file) 
        command = 'mpg123 -vC '+self.path_to_file+' '+ new_path_to_file 
        print(command)
        result = os.system(command)
        print("The RESULT is: ",result)
        if result != 0:
            raise NameError('ErrorParsingMPG123_to_system')
        
        self.info =wf.read(new_path_to_file)

    # ITERATION obj
    def __iter__(self):
        return self


    def next(self):
        return self.readSeries()


    def readSeries(self):	# return frames
        if self.info is None:
            self.initFile()
        return  self.info.readSeries(interval)    


example = "/home/juancki/Music-genre-checker/tests/SoundHelix-Song.mp3"
print("example at: ",example)

s  = Song(example)
a = [b for b in s]
print('lenght ',len(a))
