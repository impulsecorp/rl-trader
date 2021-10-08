import warnings
warnings.filterwarnings('ignore')
import os
import pickle as pkl
import numpy as np
from sklearn.preprocessing import scale, robust_scale, normalize
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
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

        self.datamatrix = DataMatrix()
        self.dts = []
        for i in range(self.datalen):
            d = DateTime(int(self.d_year[i]), int(self.d_mon[i]), int(self.d_day[i]), int(self.d_hour[i]), int(self.d_min[i]), int(self.d_sec[i]))
            self.dts.append(d)
            self.datamatrix.add_row(d, float(self.d_open[i]), float(self.d_high[i]), float(self.d_low[i]), float(self.d_close[i]), int(self.d_volume[i]))

        self.prs = []
        self.frs = []

    def process(self):
        # The raw data (price range) + time of day (fixed range)
        self.prs.append(self.d_open)
        self.prs.append(self.d_high)
        self.prs.append(self.d_low)
        self.prs.append(self.d_close)

        t = ((self.d_hour * 60 + self.d_min) - 240) / 960
        self.frs.append(t)

        self.prs.append(ta.MA(self.d_open, timeperiod=7))
        self.prs.append(ta.MA(self.d_high, timeperiod=7))
        self.prs.append(ta.MA(self.d_low,  timeperiod=7))
        self.prs.append(ta.MA(self.d_close, timeperiod=7))
        
        """

        # ARIMA
        arima = sm.tsa.statespace.SARIMAX(self.d_close, order=(7, 1, 7), seasonal_order=(0, 0, 0, 0),
                                          enforce_stationarity=False, enforce_invertibility=False, ).fit()
        arima.summary()
        self.prs.append(arima.predict())

        # SARIMA
        sarima = sm.tsa.statespace.SARIMAX(self.d_close, order=(7, 1, 7), seasonal_order=(7, 1, 7, 1),
                                           enforce_stationarity=False, enforce_invertibility=False).fit()
        sarima.summary()
        self.prs.append(sarima.predict())

        # ARIMAX
        arimax = sm.tsa.statespace.SARIMAX(self.d_close, order=(7, 1, 7), seasonal_order=(0, 0, 0, 0),  # exog = exog_train,
                                           enforce_stationarity=False, enforce_invertibility=False, ).fit()
        arimax.summary()
        self.prs.append(arimax.predict())

        # SARIMAX
        sarimax = sm.tsa.statespace.SARIMAX(self.d_close, order=(7, 1, 7), seasonal_order=(1, 0, 5, 1),  # exog = exog_train,
                                            enforce_stationarity=False, enforce_invertibility=False).fit()
        sarimax.summary()
        self.prs.append(sarimax.predict())
        """


        # In[381]:
        bb_upperband, bb_middleband, bb_lowerband = ta.BBANDS(self.d_close, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
        self.prs.append(bb_upperband)
        self.prs.append(bb_middleband)
        self.prs.append(bb_lowerband)

        # In[382]:
        bb_upperband, bb_middleband, bb_lowerband = ta.BBANDS(self.d_close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.prs.append(bb_upperband)
        self.prs.append(bb_middleband)
        self.prs.append(bb_lowerband)

        # In[383]:
        bb_upperband, bb_middleband, bb_lowerband = ta.BBANDS(self.d_close, timeperiod=5, nbdevup=3, nbdevdn=3, matype=0)
        self.prs.append(bb_upperband)
        self.prs.append(bb_middleband)
        self.prs.append(bb_lowerband)

        # In[384]:
        bb_upperband, bb_middleband, bb_lowerband = ta.BBANDS(self.d_close, timeperiod=14, nbdevup=3, nbdevdn=3, matype=0)
        self.prs.append(bb_upperband)
        self.prs.append(bb_middleband)
        self.prs.append(bb_lowerband)

        # In[385]:
        Double_EMA = ta.DEMA(self.d_close, timeperiod=60)
        self.prs.append(Double_EMA)

        # In[386]:
        EMA = ta.EMA(self.d_close, timeperiod=60)
        self.prs.append(EMA)

        # In[387]:
        EMA = ta.EMA(self.d_close, timeperiod=10)
        self.prs.append(EMA)

        # In[388]:
        KAMA = ta.KAMA(self.d_close, timeperiod=60)
        self.prs.append(KAMA)

        # In[389]:
        KAMA = ta.KAMA(self.d_close, timeperiod=10)
        self.prs.append(KAMA)

        # In[390]:
        MA = ta.MA(self.d_close, timeperiod=15, matype=0)
        self.prs.append(MA)

        # In[391]:
        MA = ta.MA(self.d_close, timeperiod=60, matype=0)
        self.prs.append(MA)

        # In[399]:
        TEMA = ta.TEMA(self.d_close, timeperiod=60)
        self.prs.append(TEMA)

        # In[400]:
        TRIMA = ta.TRIMA(self.d_close, timeperiod=60)
        self.prs.append(TRIMA)

        # In[401]:
        WMA = ta.WMA(self.d_close, timeperiod=7)
        self.prs.append(WMA)

        # In[404]:
        TEMA = ta.TEMA(self.d_close, timeperiod=7)
        self.prs.append(TEMA)

        # In[405]:
        TRIMA = ta.TRIMA(self.d_close, timeperiod=7)
        self.prs.append(TRIMA)

        # In[406]:
        WMA = ta.WMA(self.d_close, timeperiod=7)
        self.prs.append(WMA)

        # In[394]:
        MIDPOINT = ta.MIDPOINT(self.d_close, timeperiod=7)
        self.prs.append(MIDPOINT)

        # In[395]:
        MIDPRICE = ta.MIDPRICE(self.d_high, self.d_low, timeperiod=7)
        self.prs.append(MIDPRICE)

        # In[396]:
        SAR = ta.SAR(self.d_high, self.d_low, acceleration=.05, maximum=.5)
        self.prs.append(SAR)

        # In[397]:
        SAREXT = ta.SAREXT(self.d_high, self.d_low, startvalue=0, offsetonreverse=0, accelerationinitlong=.05,
                           accelerationlong=.05,
                           accelerationmaxlong=.5, accelerationinitshort=.05, accelerationshort=.05,
                           accelerationmaxshort=.5)
        self.frs.append(SAREXT)

        # In[398]:
        T3 = ta.T3(self.d_close, timeperiod=10, vfactor=.14)
        self.prs.append(T3)

        # In[409]:
        T3 = ta.T3(self.d_close, timeperiod=20, vfactor=.07)
        self.prs.append(T3)

        # In[402]:
        ADX = ta.ADX(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(ADX)

        # In[403]:
        ADXR = ta.ADXR(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(ADXR)


        # In[410]:
        APO = ta.APO(self.d_close, fastperiod=6, slowperiod=13, matype=0)
        self.frs.append(APO)

        # In[411]:
        AROONDOWN, AROONUP = ta.AROON(self.d_high, self.d_low, timeperiod=7)
        self.frs.append(AROONDOWN)
        self.frs.append(AROONUP)

        # In[412]:
        AROONDOWN, AROONUP = ta.AROON(self.d_high, self.d_low, timeperiod=28)
        self.frs.append(AROONDOWN)
        self.frs.append(AROONUP)

        # In[413]:
        AROONOSC = ta.AROONOSC(self.d_high, self.d_low, timeperiod=7)
        self.frs.append(AROONOSC)

        # In[414]:
        AROONOSC = ta.AROONOSC(self.d_high, self.d_low, timeperiod=28)
        self.frs.append(AROONOSC)

        # In[415]:
        CCI = ta.CCI(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(CCI)

        # In[416]:
        CCI = ta.CCI(self.d_high, self.d_low, self.d_close, timeperiod=28)
        self.frs.append(CCI)

        # In[417]:
        CMO = ta.CMO(self.d_close, timeperiod=14)
        self.frs.append(CMO)

        # In[418]:
        CMO = ta.CMO(self.d_close, timeperiod=7)
        self.frs.append(CMO)

        # In[419]:
        CMO = ta.CMO(self.d_close, timeperiod=28)
        self.frs.append(CMO)

        # In[420]:
        DX = ta.DX(self.d_high, self.d_low, self.d_close, timeperiod=14)
        self.frs.append(CMO)

        # In[421]:
        DX = ta.DX(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(DX)

        # In[422]:
        DX = ta.DX(self.d_high, self.d_low, self.d_close, timeperiod=28)
        self.frs.append(DX)

        # In[423]:
        macd, macdsignal, macdhist = ta.MACD(self.d_close, fastperiod=6, slowperiod=13, signalperiod=5)
        self.frs.append(macd)
        self.frs.append(macdsignal)
        self.frs.append(macdhist)

        # In[424]:
        macd, macdsignal, macdhist = ta.MACDEXT(self.d_close, fastperiod=6, fastmatype=0, slowperiod=13, slowmatype=0,
                                                signalperiod=5, signalmatype=0)
        self.frs.append(macd)
        self.frs.append(macdsignal)
        self.frs.append(macdhist)

        # In[425]:
        macd, macdsignal, macdhist = ta.MACDFIX(self.d_close, signalperiod=5)
        self.frs.append(macd)
        self.frs.append(macdsignal)
        self.frs.append(macdhist)

        # In[426]:
        macd, macdsignal, macdhist = ta.MACD(self.d_close, fastperiod=20, slowperiod=40, signalperiod=9)
        self.frs.append(macd)
        self.frs.append(macdsignal)
        self.frs.append(macdhist)

        # In[431]:
        MINUS_DI = ta.MINUS_DI(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(MINUS_DI)

        # In[432]:
        MINUS_DM = ta.MINUS_DM(self.d_high, self.d_low, timeperiod=14)
        self.frs.append(MINUS_DM)

        # In[433]:
        MOM = ta.MOM(self.d_close, timeperiod=10)
        self.frs.append(MOM)

        # In[434]:
        MOM = ta.MOM(self.d_close, timeperiod=5)
        self.frs.append(MOM)

        # In[438]:
        PLUS_DI = ta.PLUS_DI(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(PLUS_DI)

        # In[439]:
        PLUS_DM = ta.PLUS_DM(self.d_high, self.d_low, timeperiod=7)
        self.frs.append(PLUS_DM)

        # In[440]:
        PPO = ta.PPO(self.d_close, fastperiod=6, slowperiod=13, matype=0)
        self.frs.append(PPO)

        # In[441]:
        ROC = ta.ROC(self.d_close, timeperiod=5)
        self.frs.append(ROC)

        # In[442]:
        ROC = ta.ROC(self.d_close, timeperiod=20)
        self.frs.append(ROC)

        # In[445]:
        ROCP = ta.ROCR(self.d_close, timeperiod=5)
        self.frs.append(ROCP)

        # In[446]:
        ROCR100 = ta.ROCR100(self.d_close, timeperiod=5)
        self.frs.append(ROCR100)

        # In[447]:
        ROCP = ta.ROCR(self.d_close, timeperiod=20)
        self.frs.append(ROCP)

        # In[448]:
        ROCR100 = ta.ROCR100(self.d_close, timeperiod=20)
        self.frs.append(ROCR100)

        # In[449]:
        RSI = ta.RSI(self.d_close, timeperiod=7)
        self.frs.append(RSI)

        # In[450]:
        RSI = ta.RSI(self.d_close, timeperiod=20)
        self.frs.append(RSI)

        # In[455]:
        SLOWK, SLOWD = ta.STOCH(self.d_high, self.d_low, self.d_close, fastk_period=10, slowk_period=5, slowk_matype=0, slowd_period=5,
                                slowd_matype=0)
        self.frs.append(SLOWK)
        self.frs.append(SLOWD)

        # In[456]:
        FASTK, FASTD = ta.STOCHF(self.d_high, self.d_low, self.d_close, fastk_period=10, fastd_period=5, fastd_matype=0)
        self.frs.append(FASTK)
        self.frs.append(FASTD)

        # In[460]:
        TRIX = ta.TRIX(self.d_close, timeperiod=10)
        self.frs.append(TRIX)

        # In[461]:
        ULTOSC = ta.ULTOSC(self.d_high, self.d_low, self.d_close, timeperiod1=14, timeperiod2=28, timeperiod3=60)
        self.frs.append(ULTOSC)

        # In[462]:
        WILLR = ta.WILLR(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(WILLR)

        # In[463]:
        WILLR = ta.WILLR(self.d_high, self.d_low, self.d_close, timeperiod=28)
        self.frs.append(WILLR)

        # In[467]:
        ATR = ta.ATR(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(ATR)

        # In[468]:
        ATR = ta.ATR(self.d_high, self.d_low, self.d_close, timeperiod=28)
        self.frs.append(ATR)

        # In[472]:
        NATR = ta.NATR(self.d_high, self.d_low, self.d_close, timeperiod=7)
        self.frs.append(NATR)

        # In[479]:
        BETA = ta.BETA(self.d_high, self.d_low, timeperiod=10)
        self.frs.append(BETA)

        # In[480]:
        BETA = ta.BETA(self.d_high, self.d_low, timeperiod=20)
        self.frs.append(BETA)

        # In[483]:
        CORREL = ta.CORREL(self.d_high, self.d_low, timeperiod=15)
        self.frs.append(CORREL)

        # In[484]:
        LINEARREG = ta.LINEARREG(self.d_close, timeperiod=7)
        self.prs.append(LINEARREG)

        # In[485]:
        LINEARREG_ANGLE = ta.LINEARREG_ANGLE(self.d_close, timeperiod=7)
        self.frs.append(LINEARREG_ANGLE)

        # In[486]:
        LINEARREG_INTERCEPT = ta.LINEARREG_INTERCEPT(self.d_close, timeperiod=7)
        self.prs.append(LINEARREG_INTERCEPT)

        # In[487]:
        LINEARREG_SLOPE = ta.LINEARREG_SLOPE(self.d_close, timeperiod=7)
        self.frs.append(LINEARREG_SLOPE)

        # In[488]:
        STDDEV = ta.STDDEV(self.d_close, timeperiod=10, nbdev=2)
        self.frs.append(STDDEV)

        # In[495]:
        TSF = ta.TSF(self.d_close, timeperiod=7)
        self.prs.append(TSF)

        # In[496]:
        VAR = ta.VAR(self.d_close, timeperiod=10, nbdev=1)
        self.frs.append(VAR)

        # In[498]:
        MAX = ta.MAX(self.d_close, timeperiod=15)
        self.prs.append(MAX)

        # In[500]:
        MIN = ta.MIN(self.d_close, timeperiod=15)
        self.prs.append(MIN)


        intrabar_percent = """((Price('high') - Price('close')) /
              (Price('high') - Price('low') + F(0.000000001)) * F(100.0))
        """

        some_percent = """ (PRICE('high') - PRICE('low')) / IFTHENELSE( ABS(PRICE('open') - (PRICE('close'))) != F(0.0),
                                                                        ABS(PRICE('open') - (PRICE('close'))),
                                                                        F(1.0) ) """

        prev_opposite_bar = """(

        ((PRICE('close', bars_back=1) > PRICE('open', bars_back=1)) and (PRICE('close') < PRICE('open')))

        or

        ((PRICE('close', bars_back=1) < PRICE('open', bars_back=1)) and (PRICE('close') > PRICE('open')))

        )"""

        prev_same_bar = """(

        ((PRICE('close', bars_back=1) > PRICE('open', bars_back=1)) and (PRICE('close') > PRICE('open')))

        or

        ((PRICE('close', bars_back=1) < PRICE('open', bars_back=1)) and (PRICE('close') < PRICE('open')))

        )"""

        new_day = """NEW_DAY"""

        at_open = """TIME_IS(09:31)"""

        in_regular_hours = """TIME_IS_BETWEEN(09:31, 16:00)"""

        pmove = """(((PRICE('close', data={0}) -
                       VALUE_AT_TIME(PRICE('open', data={0}), ['open_time',09:31],
                       days_ago=['days_ago', 0], reset_overnight=true)) /
                       VALUE_AT_TIME(PRICE('open', data={0}), ['open_time',09:31],
                       days_ago=['days_ago', 0], reset_overnight=true))*F(100.0))""".format(0)

        pmove2 = """(((PRICE('close', data={0}) -
                       VALUE_AT_TIME(PRICE('open', data={0}), ['open_time',04:00],
                       days_ago=['days_ago', 0], reset_overnight=true)) /
                       VALUE_AT_TIME(PRICE('open', data={0}), ['open_time',04:00],
                       days_ago=['days_ago', 0], reset_overnight=true))*F(100.0))""".format(0)

        strend_up = "SUPERTREND_UP(1.5, 3, 3)"
        strend_down = "SUPERTREND_DOWN(1.5, 3, 3)"

        up_bars_in_row = "TIMES_IN_ROW(PRICE('open') < PRICE('close'))"
        down_bars_in_row = "TIMES_IN_ROW(PRICE('open') > PRICE('close'))"

        candles = [['CDL_2CROWS', '2 Crows'],
                   ['CDL_3BLACKCROWS', '3 Black Crows'],
                   ['CDL_3INSIDE', '3 Inside'],
                   ['CDL_3LINESTRIKE', '3 Line Strike'],
                   ['CDL_3OUTSIDE', '3 Outside'],
                   ['CDL_3STARSINSOUTH', '3 Stars In South'],
                   ['CDL_3WHITESOLDIERS', '3 White Soldiers'],
                   ['CDL_ADVANCEBLOCK', 'Advance Block'],
                   ['CDL_BELTHOLD', 'Belt Hold'],
                   ['CDL_BREAKAWAY', 'Break Away'],
                   ['CDL_CLOSINGMARUBOZU', 'Closing Maribozu'],
                   ['CDL_CONCEALBABYSWALL', 'Conceal Babys Wall'],
                   ['CDL_COUNTERATTACK', 'Counter Attack'],
                   ['CDL_DOJI', 'Doji'],
                   ['CDL_DOJISTAR', 'Doji Star'],
                   ['CDL_DRAGONFLYDOJI', 'Dragonfly Doji'],
                   ['CDL_ENGULFING', 'Engulfing Pattern'],
                   ['CDL_GAPSIDESIDEWHITE', 'Gap Side By Side White'],
                   ['CDL_GRAVESTONEDOJI', 'Gravestone Doji'],
                   ['CDL_HAMMER', 'Hammer'],
                   ['CDL_HANGINGMAN', 'Hanging Man'],
                   ['CDL_HARAMI', 'Harami Pattern'],
                   ['CDL_HARAMICROSS', 'Harami Cross Pattern'],
                   ['CDL_HIGHWAVE', 'High-Wave Candle'],
                   ['CDL_HIKKAKE', 'Hikkake Pattern'],
                   ['CDL_HIKKAKEMOD', 'Modified Hikkake Pattern'],
                   ['CDL_HOMINGPIGEON', 'Homing Pigeon'],
                   ['CDL_IDENTICAL3CROWS', 'Identical 3 Crows'],
                   ['CDL_INNECK', 'In Neck Pattern'],
                   ['CDL_INVERTEDHAMMER', 'Inverted Hammer'],
                   ['CDL_KICKING', 'Kicking'],
                   ['CDL_KICKINGBYLENGTH', 'Kicking By Length'],
                   ['CDL_LADDERBOTTOM', 'Ladder Bottom'],
                   ['CDL_LONGLEGGEDDOJI', 'Long Legged Doji'],
                   ['CDL_LONGLINE', 'Long Line'],
                   ['CDL_MARUBOZU', 'Maribozu'],
                   ['CDL_MATCHINGLOW', 'Matching Low'],
                   ['CDL_ONNECK', 'On Neck Pattern'],
                   ['CDL_PIERCING', 'Piercing Pattern'],
                   ['CDL_RICKSHAWMAN', 'Rickshaw Man'],
                   ['CDL_RISEFALL3METHODS', 'Rising Falling 3 Methods'],
                   ['CDL_SEPARATINGLINES', 'Separating Lines'],
                   ['CDL_SHOOTINGSTAR', 'Shooting Star'],
                   ['CDL_SHORTLINE', 'Short Line'],
                   ['CDL_SPINNINGTOP', 'Spinning Top'],
                   ['CDL_STALLEDPATTERN', 'Stalled Pattern'],
                   ['CDL_STICKSANDWICH', 'Stick Sandwich'],
                   ['CDL_TAKURI', 'Takuri'],
                   ['CDL_TASUKIGAP', 'Tasuki Gap'],
                   ['CDL_THRUSTING', 'Thrusting Pattern'],
                   ['CDL_TRISTAR', 'Tristar Pattern'],
                   ['CDL_UNIQUE3RIVER', 'Unique 3 River'],
                   ['CDL_UPSIDEGAP2CROWS', 'Upside Gap 2 Crows'],
                   ['CDL_XSIDEGAP3METHODS', 'Side Gap 3 Methods']]


        signals = [(intrabar_percent, 100.0),
                   (some_percent, 1.0),
                   (prev_opposite_bar, 1.0),
                   (prev_same_bar, 1.0),
                   (at_open, 1.0),
                   (new_day, 1.0),
                   (in_regular_hours, 1.0),
                   (pmove, 1.0),
                   (pmove2, 1.0),
                   (strend_up, 1.0),
                   (strend_down, 1.0),
                   (up_bars_in_row, 5.0),
                   (down_bars_in_row, 5.0), ]

        signals += [(x[0], 1.0) for x in candles]

        # All TradeSys signal tree based indicators
        for s, mx in signals:
            a = np.array(get_stree_log(s, self.datamatrix), dtype=np.float32)
            a = a / mx
            self.frs.append(a)

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
        pr_data = scale(pr_data, axis=1)
        
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

        fr_data = np.vstack(self.frs)
        b = fr_data[:, 1:] - fr_data[:, 0:-1]
        b = np.hstack([np.zeros(b.shape[0]).reshape(-1,1), b])
        fr_data = b
        fr_data = scale(fr_data, axis=1)
        
        
        self.data = np.vstack([pr_data, fr_data])

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