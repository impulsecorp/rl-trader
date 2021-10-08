import warnings
warnings.filterwarnings('ignore')
import os
import pickle as pkl
import numpy as np
from sklearn.preprocessing import scale, robust_scale, normalize
import talib as ta
try:
    from tradesys import *
except ImportError:
    import sys
    sys.path.insert(0, '/home/peter/code/projects/tradesys/')
    from tradesys import *

class DataModel:
    
    def __init__(self):
        self.l = np.load('x_log_spy.npy')
        self.i = 0

    def input(self, timewin):
        pass

    def process(self):
        pass

    def cut_nans(self):
        return 0

    def cut_nans_fast(self, cutidx):
        pass

    def output(self, winlen):
        x = self.l[self.i]
        self.i += 1
        return x