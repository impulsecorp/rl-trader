import warnings
warnings.filterwarnings('ignore')
import os
import pickle as pkl
import numpy as np
from sklearn.preprocessing import scale, robust_scale, normalize
import talib as ta
try:
    from tradesys import *
    import tradesys.ml as ml
except ImportError:
    import sys
    sys.path.insert(0, '/home/peter/code/projects/tradesys/')
    from tradesys import *
    import tradesys.ml as ml

class DataModel(ml.BaseDataModel):

    def input(self, timewin):

        # take the raw timewin and construct a DataMatrix from it
        self.datalen = timewin.shape[1]
        self.d_open = timewin[0, :]
        self.d_high = timewin[1, :]
        self.d_low = timewin[2, :]
        self.d_close = timewin[3, :]
        self.d_volume = timewin[4, :]
        self.d_year = timewin[5, :]
        self.d_mon = timewin[6, :]
        self.d_day = timewin[7, :]
        self.d_hour = timewin[8, :]
        self.d_min = timewin[9, :]
        self.d_sec = timewin[10, :]

        #self.datamatrix = DataMatrix()
        #self.dts = []
        #for i in range(self.datalen):
        #    d = DateTime(int(self.d_year[i]), int(self.d_mon[i]), int(self.d_day[i]), int(self.d_hour[i]), int(self.d_min[i]), int(self.d_sec[i]))
        #    self.dts.append(d)
        #    self.datamatrix.add_row(d, float(self.d_open[i]), float(self.d_high[i]), float(self.d_low[i]), float(self.d_close[i]), int(self.d_volume[i]))

        self.prs = []
        self.frs = []

    def process(self):
        # The raw data (price range) + time of day (fixed range)
        #self.prs.append(self.d_open)
        #self.prs.append(self.d_high)
        #self.prs.append(self.d_low)
        self.prs.append(self.d_close)

        #t = ((self.d_hour * 60 + self.d_min) - 240) / 960
        #self.frs.append(t)

        #self.prs.append(ta.MA(self.d_open, timeperiod=7))
        #self.prs.append(ta.MA(self.d_high, timeperiod=7))
        #self.prs.append(ta.MA(self.d_low,  timeperiod=7))
        #self.prs.append(ta.MA(self.d_close, timeperiod=7))


        
        # Stack and scale all data
        pr_data = np.vstack(self.prs)
        
        # lag the data
        b = pr_data[:, 1:] - pr_data[:, 0:-1]
        b = np.hstack([np.zeros(b.shape[0]).reshape(-1,1), b])
        pr_data = b

        # scale the price data and all price range indicators along with it
        #mnp, mxp = np.min(pr_data[3]), np.max(pr_data[3])
        #pr_data = (((pr_data - mnp) / abs(mxp - mnp)) - 0.5) * 2
        #self.lag_close_minmax = [mnp, mxp]
        #pr_data = pr_data * 25.0
        #pr_data = scale(pr_data, axis=1)
        
        self.minmaxes = []

        # scale each row of the fixed range indicators separately
        #for i, r in enumerate(self.frs):
        #    a = r.copy()
        #    a[np.isnan(a)] = 0
        #    mn, mx = np.min(a), np.max(a)
        #    self.minmaxes.append( [mn, mx] )
        #    if mx - mn > 0:
        #        self.frs[i] = (((r - mn) / abs(mx - mn)) - 0.5) * 2
        #    else:
        #        self.frs[i] = np.zeros_like(self.frs[i])

        #fr_data = np.vstack(self.frs)
        #b = fr_data[:, 1:] - fr_data[:, 0:-1]
        #b = np.hstack([np.zeros(b.shape[0]).reshape(-1,1), b])
        #fr_data = b
        #fr_data = scale(fr_data, axis=1)
        
        self.data = scale(np.vstack([pr_data]), axis=1)#, fr_data])

    def cut_nans(self):
        # find the real start of the data (without NaN)
        nc = 0
        for cl in range(self.data.shape[1]):
            a = self.data[:, cl]
            if len(a[np.isnan(a)]) > 0:
                nc = cl
        # cut off the parts that contain NaN
        self.data = self.data[:, nc + 1:]
        return nc+1

    def cut_nans_fast(self, cutidx):
        self.data = self.data[:, cutidx:]

    def output(self, winlen):
        x = self.data[:, -winlen:]
        return x#scale(x)