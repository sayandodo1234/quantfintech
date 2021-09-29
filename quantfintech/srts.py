import pandas as pd
import datetime as dt
import os
import numpy as np
import sys
import statsmodels.api as sm
from . import assist as at
from .Security_init import Security
# srts - support resistance trend slope
############ Support Resistance Pivot ###################

class Levels(Security):
    def __init__(self,ohlc_df):
        Security.__init__(self,ohlc_df)
        self.levels_dict = {}
        self.high = self.ohlc_df["high"][-1]
        self.low = self.ohlc_df["low"][-1]
        self.close = self.ohlc_df["close"][-1]
        self.pivot = (self.high + self.low + self.close)/3
    
   
    def support(self,num_support=3):
        if (num_support <1) or (num_support>3):
            sys.exit("Number of support lines must be between 1 and 3 inclusive")
        s1 = 2*self.pivot - self.high
        self.levels_dict["Support1"] = s1
        if num_support>1:
            s2 = self.pivot  - (self.high - self.low)
            self.levels_dict["Support2"] = s2
        if num_support == 3:
            s3 = self.low - 2*(self.high - self.pivot)
            self.levels_dict["Support3"] = s3
        return
    
    def resistance(self,num_resistance = 3):
        if (num_resistance <1) or (num_resistance>3):
            sys.exit("Number of resistance lines must be between 1 and 3 inclusive")
        
        r1 = 2*self.pivot - self.low
        self.levels_dict["Resistance1"] = r1
        if num_resistance>1:
            r2 = self.pivot  + (self.high - self.low)
            self.levels_dict["Resistance2"] = r2
        if num_resistance == 3:
            r3 = self.high + 2*(self.pivot - self.low)
            self.levels_dict["Resistance3"] = r3
        return 
    
 
    
    def compute(self,num_supp_res=3,ret_dict = True):
        self.levels_dict["Pivot"] = self.pivot
        self.support(num_supp_res)
        self.resistance(num_supp_res)
        if  ret_dict == True:
            return self.levels_dict
        return
    
    def show(self,num_supp_res=3):
        self.compute_all(num_supp_res,False)
        print(self.levels_dict)
        return
    
    
############ Slope Trend #########################
class Slope(Security):
    
    def __init__(self,ohlc_df,points):
        Security.__init__(self,ohlc_df)
        self.points = points
    
    def calculate(self):
        "function to calculate the slope of regression line for n consecutive points on a plot"
        df = self.ohlc_df.iloc[-1*self.points:,:]
        y = ((df["open"] + df["close"])/2).values
        x = np.array(range(self.points))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slope = np.rad2deg(np.arctan(results.params[-1]))
        return slope
    
class Trend(Security):
    
    def __init__(self,ohlc_df,points,percent_ratio=0.7):
        Security.__init__(self,ohlc_df)
        self.points = points
        self.percent_ratio = percent_ratio
        
    
    def show(self):
        "function to assess the trend by analyzing each candle"
        df = self.ohlc_df.copy()
        df["up"] = np.where(df["low"]>=df["low"].shift(1),1,0)
        df["dn"] = np.where(df["high"]<=df["high"].shift(1),1,0)
        if df["close"][-1] > df["open"][-1]:
            if df["up"][-1*self.points:].sum() >= self.percent_ratio*self.points:
                return "uptrend"
        elif df["open"][-1] > df["close"][-1]:
            if df["dn"][-1*self.points:].sum() >= self.percent_ratio*self.points:
                return "downtrend"
        else:
            return "None"
        #show(self.points,self.percent_ratio)