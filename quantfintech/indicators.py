#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:15:20 2021

@author: sayandebhowmick
"""

import pandas as pd
import datetime as dt
import os
import numpy as np
import sys
import statsmodels.api as sm
#import assist as at
from . import assist as at

class Security:
    

    def __init__(self, security_df):
        self.ohlc_df = security_df.copy(deep=True)
        self.ohlc_cols = ['open', 'high', 'low', 'close','volume']
        self.process()
            

    def process(self):
        self.ohlc_df.rename(columns = {"Open":"open", "Close":"close","Volume":"volume","High":"high","Low":"low"},inplace = True)
        if pd.Series(self.ohlc_cols).isin(self.ohlc_df.columns).all():
            self.ohlc_df = self.ohlc_df[self.ohlc_cols]
        else:
            
            commomCols_set = set(self.ohlc_cols) & set(self.ohlc_df.columns)
            #commonCols_list_temp = list(commomCols_set)
            #commonCols_list = [cols.lower() for cols in commonCols_list_temp]
            missingCols_list = list(set(self.ohlc_cols) - commomCols_set)
            print("Missing columns : {0}".format(missingCols_list))
            sys.exit("Terminating")


class RSI(Security):
    
    def __init__(self,ohlc_df,window, exponential_moving=True,decay_type="com",decay_value=-1.0,alpha_val=-1):
        Security.__init__(self,ohlc_df)
        self.window = window
        self.exponential_moving = exponential_moving
        self.decay_type = decay_type
        self.decay_value = decay_value
        self.alpha_val = alpha_val


    def compute(self):
        "function to calculate RSI"
        df = self.ohlc_df.copy(deep=True)
        delta = df["close"].diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        # first value is average of gains
        u[u.index[self.window-1]] = np.mean(u[:self.window])
        u = u.drop(u.index[:(self.window-1)])
        # first value is average of losses
        d[d.index[self.window-1]] = np.mean(d[:self.window])
        d = d.drop(d.index[:(self.window-1)])
        if self.exponential_moving == True:
            at.exponential_moving_parameter_check(self.decay_type,self.alpha_val)
            num = at.compute_exponential_moving(u,self.window,self.decay_type,self.decay_value,self.alpha_val).mean()
            dem = at.compute_exponential_moving(d,self.window,self.decay_type,self.decay_value,self.alpha_val).mean()
            rs = num/dem

        else:
            rs = u.rolling(self.window).mean()/d.rolling(self.window).mean()
        df["rsi"] = 100 - 100 / (1+rs)
        return df

class ATR(Security):
    
    def __init__(self,ohlc_df,window, exponential_moving=True,decay_type="com",decay_value=-1.0,alpha_val=-1):
        Security.__init__(self,ohlc_df)
        self.window = window
        self.exponential_moving = exponential_moving
        self.decay_type = decay_type
        self.decay_value = decay_value
        self.alpha_val = alpha_val

    def compute(self):

        df = self.ohlc_df.copy(deep=True)
        df['H-L'] = abs(df['high']-df['low'])
        df['H-PC'] = abs(df['high']-df['close'].shift(1))
        df['L-PC'] = abs(df['low']-df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
        if self.exponential_moving == True:
            at.exponential_moving_parameter_check(self.decay_type,self.alpha_val)
            df['ATR'] = at.compute_exponential_moving(df['TR'],self.window,self.decay_type,self.decay_value,self.alpha_val).mean()
        else:
            df['ATR'] = df['TR'].rolling(self.window).mean()


        return df


class BollingerBand(Security):
    
    
    def __init__(self,ohlc_df,window,multiplier):
        Security.__init__(self,ohlc_df)
        self.window = window
        self.multiplier = multiplier
    

    def compute(self):
        "function to calculate Bollinger Band"
        df = self.ohlc_df.copy(deep=True)
        df["MA"] = df['close'].rolling(self.window).mean()
        #df["MA"] = df['close'].ewm(span=n,min_periods=n).mean()
        # ddof=0 is required since we want to take the standard deviation of the population and not sample
        df["BB_up"] = df["MA"] + self.multiplier * \
        df['close'].rolling(self.window).std(ddof=0)
        # ddof=0 is required since we want to take the standard deviation of the population and not sample
        df["BB_dn"] = df["MA"] - self.multiplier * \
        df['close'].rolling(self.window).std(ddof=0)
        df["BB_width"] = df["BB_up"] - df["BB_dn"]
        df.dropna(inplace=True)

        return df

class MACD(Security):
    
    def __init__(self,ohlc_df,ma_fast_window=12,ma_slow_window=26,ma_signal_window=9,exponential_moving=True,decay_type="span",decay_value=-1.0,alpha_val=-1):
        Security.__init__(self,ohlc_df)
        self.ma_fast_window= ma_fast_window
        self.ma_slow_window=ma_slow_window
        self.ma_signal_window = ma_signal_window
        self.exponential_moving = exponential_moving
        self.decay_type = decay_type
        self.decay_value = decay_value
        self.alpha_val = alpha_val
        
    def compute(self):
        """function to calculate MACD
       typical values a(fast moving average) = 12; 
                      b(slow moving average) =26; 
                      c(signal line ma window) =9"""
        
        df = self.ohlc_df.copy(deep=True)
        if self.exponential_moving == True:
            at.exponential_moving_parameter_check(self.decay_type,self.alpha_val)
            df["MA_Fast"] = at.compute_exponential_moving(df["close"],self.ma_fast_window,self.decay_type,self.decay_value,self.alpha_val).mean()
            #df["MA_Fast"]=df["close"].ewm(span=ma_fast_window,min_periods=ma_fast_window).mean()
            df["MA_Slow"] = at.compute_exponential_moving(df["close"],self.ma_slow_window,self.decay_type,self.decay_value,self.alpha_val).mean()
            #df["MA_Slow"]=df["close"].ewm(span=ma_slow_window,min_periods=ma_slow_window).mean()
            df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
            df["Signal"]=at.compute_exponential_moving(df["MACD"],self.ma_signal_window,self.decay_type,self.decay_value,self.alpha_val).mean()
            #df["Signal"]=df["MACD"].ewm(span=ma_signal_window,min_periods=ma_signal_window).mean()
        else:
            df["MA_Fast"]=df["close"].rolling(self.ma_fast_window).mean()
            df["MA_Slow"]=df["close"].rolling(self.ma_slow_window).mean()
            df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
            df["Signal"]=df["MACD"].rolling(self.ma_signal_window).mean()
            

        df.dropna(inplace=True)
        return df
    
class ADX(Security):
    
    def __init__(self,ohlc_df,window):
        Security.__init__(self,ohlc_df)
        self.window = window
    
    def compute(self):
        "function to calculate ADX"
        df2 = self.ohlc_df.copy(deep=True)
        df2['H-L']=abs(df2['high']-df2['low'])
        df2['H-PC']=abs(df2['high']-df2['close'].shift(1))
        df2['L-PC']=abs(df2['low']-df2['close'].shift(1))
        df2['TR']=df2[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
        df2['DMplus']=np.where((df2['high']-df2['high'].shift(1))>(df2['low'].shift(1)-df2['low']),df2['high']-df2['high'].shift(1),0)
        df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
        df2['DMminus']=np.where((df2['low'].shift(1)-df2['low'])>(df2['high']-df2['high'].shift(1)),df2['low'].shift(1)-df2['low'],0)
        df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
        TRn = []
        DMplusN = []
        DMminusN = []
        TR = df2['TR'].tolist()
        DMplus = df2['DMplus'].tolist()
        DMminus = df2['DMminus'].tolist()
        for i in range(len(df2)):
            if i < self.window:
                TRn.append(np.NaN)
                DMplusN.append(np.NaN)
                DMminusN.append(np.NaN)
            elif i == self.window:
                TRn.append(df2['TR'].rolling(self.window).sum().tolist()[self.window])
                DMplusN.append(df2['DMplus'].rolling(self.window).sum().tolist()[self.window])
                DMminusN.append(df2['DMminus'].rolling(self.window).sum().tolist()[self.window])
            elif i > self.window:
                TRn.append(TRn[i-1] - (TRn[i-1]/self.window) + TR[i])
                DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/self.window) + DMplus[i])
                DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/self.window) + DMminus[i])
        df2['TRn'] = np.array(TRn)
        df2['DMplusN'] = np.array(DMplusN)
        df2['DMminusN'] = np.array(DMminusN)
        df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
        df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
        df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
        df2['DIsum']=df2['DIplusN']+df2['DIminusN']
        df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
        ADX_list = []
        DX = df2['DX'].tolist()
        for j in range(len(df2)):
            if j < 2*self.window-1:
                ADX_list.append(np.NaN)
            elif j == 2*self.window-1:
                ADX_list.append(df2['DX'][j-self.window+1:j+1].mean())
            elif j > 2*self.window-1:
                ADX_list.append(((self.window-1)*ADX_list[j-1] + DX[j])/self.window)
        df2['ADX']=np.array(ADX_list)
        return df2
    
class Supertrend(Security): 
    
        
    def __init__(self,ohlc_df,window,multiplier):
        Security.__init__(self,ohlc_df)
        self.window = window
        self.multiplier = multiplier
    
    def compute(self):
        df = self.ohlc_df.copy(deep=True)
        df = ATR(df,self.window).compute_atr(self.window)
        df["B-U"]=((df['high']+df['low'])/2) + self.multiplier*df['ATR'] 
        df["B-L"]=((df['high']+df['low'])/2) - self.multiplier*df['ATR']
        df["U-B"]=df["B-U"]
        df["L-B"]=df["B-L"]
        ind = df.index
        df['Strend']=np.nan
        test = 0
        for i in range(self.window,len(df)):
            if df['close'][i-1]<=df['U-B'][i-1]:
                df.loc[ind[i],'U-B']=min(df['B-U'][i],df['U-B'][i-1])
            else: # Crossover
                df.loc[ind[i],'U-B']=df['B-U'][i]    
                
            if df['close'][i-1]>=df['L-B'][i-1]:
                df.loc[ind[i],'L-B']=max(df['B-L'][i],df['L-B'][i-1])
            else: # Crossover
                df.loc[ind[i],'L-B']=df['B-L'][i] 
    
    
        for test in range(self.window,len(df)):
            if df['close'][test-1]<=df['U-B'][test-1] and df['close'][test]>df['U-B'][test]:
                df.loc[ind[test],'Strend']=df['L-B'][test]
                break
            if df['close'][test-1]>=df['L-B'][test-1] and df['close'][test]<df['L-B'][test]:
                df.loc[ind[test],'Strend']=df['U-B'][test]
                break
            
        for i in range(test+1,len(df)):
            if df['Strend'][i-1]==df['U-B'][i-1] and df['close'][i]<=df['U-B'][i]:
                df.loc[ind[i],'Strend']=df['U-B'][i]
            elif  df['Strend'][i-1]==df['U-B'][i-1] and df['close'][i]>=df['U-B'][i]:
                df.loc[ind[i],'Strend']=df['L-B'][i]
            elif df['Strend'][i-1]==df['L-B'][i-1] and df['close'][i]>=df['L-B'][i]:
                df.loc[ind[i],'Strend']=df['L-B'][i]
            elif df['Strend'][i-1]==df['L-B'][i-1] and df['close'][i]<=df['L-B'][i]:
                df.loc[ind[i],'Strend']=df['U-B'][i]
        return df
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
    
    #def fetch_suport_resistance(self,num_supp_res = 3):
       # self.support(num_supp_res)
        #self.resistance(num_supp_res)
        
        #return 
    
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
        
       # r2 = round((pivot + (high - low)),2)
       # r3 = round((high + 2*(pivot - low)),2)
       
                
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