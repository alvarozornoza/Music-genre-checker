#/usr/bin/python3.6

# trainner.py
# @juancki
# Imports
import os


def f(path):
    if os.path.isdir(path):
        d = {}
        for name in os.listdir(path):
            d[os.path.join(path, name)] = f(os.path.join(path, name))
    else:
        d = None 
    return d


class BigFolder():
    #
    # big_folder [Example]
    #   -> classical
    #       -> violin_concerto_3_mozart.mp3
    #
    #   -> reagge 
    #       -> lo_malo.mp3
    #

    def __init__(self,path_to_folder):
        self.path_to_folder = path_to_folder
        self.dirs   = f(path_to_folder)
        self._d     =f(path_to_folder)


    def popfolder(self):
        # Returns a folder as a dict with its files as keys.
        q = self._d.keys()
        return self._d.pop(q[0]).keys()


