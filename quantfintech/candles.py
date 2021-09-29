import pandas as pd
import datetime as dt
import os
import numpy as np
import sys
import statsmodels.api as sm
from . import assist as at
from .Security_init import Security

########## Candle Pattern ###############################
class Doji(Security):
    
    def __init__(self,ohlc_df,multiplier_avg_candle_threshold=0.05,avg_candle_size_factor='median'):
        Security.__init__(self,ohlc_df)
        self.multiplier_avg_candle_threshold = multiplier_avg_candle_threshold
        self.avg_candle_size_factor = avg_candle_size_factor
    
    def search(self):
        df = self.ohlc_df.copy(deep = True)
        
        avg_candle_size = at.compute_average_candle_size(self.avg_candle_size_factor,df)
        df["doji"] = abs(df["close"] - df["open"]) <=  (self.multiplier_avg_candle_threshold * avg_candle_size)
        return df
    
class Hammer(Security):
    
    def __init__(self,ohlc_df,multiplier = 3,threshold = 0.6,err=0.001,percent_ratio = 0.1):
        Security.__init__(self,ohlc_df)
        self.multiplier = multiplier
        self.threshold = threshold
        self.err = err
        self.percent_ratio = percent_ratio
    
    def search(self):   
        """returns dataframe with hammer candle column"""
        df = self.ohlc_df.copy(deep = True)
        df["hammer"] = (((df["high"] - df["low"])>self.multiplier*(df["open"] - df["close"])) & \
                       ((df["close"] - df["low"])/(self.err + df["high"] - df["low"]) > self.threshold) & \
                       ((df["open"] - df["low"])/(self.err+ df["high"] - df["low"]) > self.threshold)) & \
                       (abs(df["close"] - df["open"]) > self.percent_ratio* (df["high"] - df["low"]))
        return df

class ShootingStar(Security):
    
    def __init__(self,ohlc_df,multiplier = 3,threshold = 0.6,err=0.001,percent_ratio = 0.1):
        Security.__init__(self,ohlc_df)
        self.multiplier = multiplier
        self.threshold = threshold
        self.err = err
        self.percent_ratio = percent_ratio
        
    def search(self):
        df = self.ohlc_df.copy()
        df["sstar"] = (((df["high"] - df["low"])>self.multiplier*(df["open"] - df["close"])) & \
                   ((df["high"] - df["close"])/(self.err + df["high"] - df["low"]) > self.threshold) & \
                   ((df["high"] - df["open"])/(self.err + df["high"] - df["low"]) > self.threshold)) & \
                   (abs(df["close"] - df["open"]) > self.percent_ratio* (df["high"] - df["low"]))
        return df

class MaruBozu(Security):
    
    def __init__(self,ohlc_df,multiplier_avg_candle=2,percent_ratio_avg_candle = 0.005,avg_candle_size_factor="median"):
        Security.__init__(self,ohlc_df)
        self.multiplier_avg_candle = multiplier_avg_candle
        self.percent_ratio_avg_candle = percent_ratio_avg_candle
        self.avg_candle_size_factor = avg_candle_size_factor
        
    def search(self):
        df = self.ohlc_df.copy()
        avg_candle_size = at.compute_average_candle_size(self.avg_candle_size_factor,df)
        df["h-c"] = df["high"]-df["close"]
        df["l-o"] = df["low"]-df["open"]
        df["h-o"] = df["high"]-df["open"]
        df["l-c"] = df["low"]-df["close"]
        df["maru_bozu"] = np.where((df["close"] - df["open"] > self.multiplier_avg_candle*avg_candle_size) & \
                                   (df[["h-c","l-o"]].max(axis=1) < self.percent_ratio_avg_candle*avg_candle_size),"maru_bozu_green",
                                   np.where((df["open"] - df["close"] > self.multiplier_avg_candle*avg_candle_size) & \
                                   (abs(df[["h-o","l-c"]]).max(axis=1) < self.percent_ratio_avg_candle*avg_candle_size),"maru_bozu_red",False))
        df.drop(["h-c","l-o","h-o","l-c"],axis=1,inplace=True)
        return df



