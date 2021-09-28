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
    

    def __init__(self, security_df,index_date = True):
        self.ohlc_df = security_df.copy(deep=True)
        if index_date == True:
            self.ohlc_cols = ['open', 'high', 'low', 'close','volume']
        else:
            self.ohlc_cols = ['date','open', 'high', 'low', 'close','volume']
        if pd.Series(self.ohlc_cols).isin(self.ohlc_df.columns).all():
            self.ohlc_df = self.ohlc_df[self.ohlc_cols]
        else:
            commomCols_set = set(self.ohlc_cols) & set(self.ohlc_df.columns)
            missingCols_list = list(set(self.ohlc_cols) - commomCols_set)
            print("Missing columns : {0}".format(missingCols_list))
            sys.exit("Terminating")


class RSI(Security):


    def compute(self, window, exponential_moving=True,decay_type="com",decay_value=-1.0,alpha_val=-1):
        "function to calculate RSI"
        df = self.ohlc_df.copy(deep=True)
        delta = df["close"].diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        # first value is average of gains
        u[u.index[window-1]] = np.mean(u[:window])
        u = u.drop(u.index[:(window-1)])
        # first value is average of losses
        d[d.index[window-1]] = np.mean(d[:window])
        d = d.drop(d.index[:(window-1)])
        if exponential_moving == True:
            at.exponential_moving_parameter_check(decay_type,alpha_val)
            num = at.compute_exponential_moving(u,window,decay_type,decay_value,alpha_val).mean()
            dem = at.compute_exponential_moving(d,window,decay_type,decay_value,alpha_val).mean()
            rs = num/dem
                 
           # rs = u.ewm(com=window, min_periods=window).mean() / \
                #d.ewm(com=window, min_periods=window).mean()
        else:
            rs = u.rolling(window).mean()/d.rolling(window).mean()
        df["rsi"] = 100 - 100 / (1+rs)
        return df


class ATR(Security):

    def compute(self, window, exponential_moving=True,decay_type="com",decay_value=-1.0,alpha_val=-1):

        df = self.ohlc_df.copy(deep=True)
        df['H-L'] = abs(df['high']-df['low'])
        df['H-PC'] = abs(df['high']-df['close'].shift(1))
        df['L-PC'] = abs(df['low']-df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
        if exponential_moving == True:
            at.exponential_moving_parameter_check(decay_type,alpha_val)
            df['ATR'] = at.compute_exponential_moving(df['TR'],window,decay_type,decay_value,alpha_val).mean()
        else:
            df['ATR'] = df['TR'].rolling(window).mean()


        return df


class BollingerBand(Security):

    def compute(self, window, multiplier):
        "function to calculate Bollinger Band"
        df = self.ohlc_df.copy(deep=True)
        df["MA"] = df['close'].rolling(window).mean()
        #df["MA"] = df['close'].ewm(span=n,min_periods=n).mean()
        # ddof=0 is required since we want to take the standard deviation of the population and not sample
        df["BB_up"] = df["MA"] + multiplier * \
        df['close'].rolling(window).std(ddof=0)
        # ddof=0 is required since we want to take the standard deviation of the population and not sample
        df["BB_dn"] = df["MA"] - multiplier * \
        df['close'].rolling(window).std(ddof=0)
        df["BB_width"] = df["BB_up"] - df["BB_dn"]
        df.dropna(inplace=True)

        return df

class MACD(Security):
    
    def compute(self,ma_fast_window=12,ma_slow_window=26,ma_signal_window=9,exponential_moving=True,decay_type="span",decay_value=-1.0,alpha_val=-1):
        """function to calculate MACD
       typical values a(fast moving average) = 12; 
                      b(slow moving average) =26; 
                      c(signal line ma window) =9"""
        
        df = self.ohlc_df.copy(deep=True)
        if exponential_moving == True:
            at.exponential_moving_parameter_check(decay_type,alpha_val)
            df["MA_Fast"] = at.compute_exponential_moving(df["close"],ma_fast_window,decay_type,decay_value,alpha_val).mean()
            #df["MA_Fast"]=df["close"].ewm(span=ma_fast_window,min_periods=ma_fast_window).mean()
            df["MA_Slow"] = at.compute_exponential_moving(df["close"],ma_slow_window,decay_type,decay_value,alpha_val).mean()
            #df["MA_Slow"]=df["close"].ewm(span=ma_slow_window,min_periods=ma_slow_window).mean()
            df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
            df["Signal"]=at.compute_exponential_moving(df["MACD"],ma_signal_window,decay_type,decay_value,alpha_val).mean()
            #df["Signal"]=df["MACD"].ewm(span=ma_signal_window,min_periods=ma_signal_window).mean()
        else:
            df["MA_Fast"]=df["close"].rolling(ma_fast_window).mean()
            df["MA_Slow"]=df["close"].rolling(ma_slow_window).mean()
            df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
            df["Signal"]=df["MACD"].rolling(ma_signal_window).mean()
            

        df.dropna(inplace=True)
        return df
    
class ADX(Security):
    
    def compute(self,window):#,decay_value=-1.0,alpha_val=-1):
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
            if i < window:
                TRn.append(np.NaN)
                DMplusN.append(np.NaN)
                DMminusN.append(np.NaN)
            elif i == window:
                TRn.append(df2['TR'].rolling(window).sum().tolist()[window])
                DMplusN.append(df2['DMplus'].rolling(window).sum().tolist()[window])
                DMminusN.append(df2['DMminus'].rolling(window).sum().tolist()[window])
            elif i > window:
                TRn.append(TRn[i-1] - (TRn[i-1]/window) + TR[i])
                DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/window) + DMplus[i])
                DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/window) + DMminus[i])
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
            if j < 2*window-1:
                ADX_list.append(np.NaN)
            elif j == 2*window-1:
                ADX_list.append(df2['DX'][j-window+1:j+1].mean())
            elif j > 2*window-1:
                ADX_list.append(((window-1)*ADX_list[j-1] + DX[j])/window)
        df2['ADX']=np.array(ADX_list)
        return df2
    
class Supertrend(Security): 
    
    def compute(self,window,multiplier):
        df = self.ohlc_df.copy(deep=True)
        df = ATR(df,window).compute_atr(window)
        df["B-U"]=((df['high']+df['low'])/2) + multiplier*df['ATR'] 
        df["B-L"]=((df['high']+df['low'])/2) - multiplier*df['ATR']
        df["U-B"]=df["B-U"]
        df["L-B"]=df["B-L"]
        ind = df.index
        df['Strend']=np.nan
        test = 0
        for i in range(window,len(df)):
            if df['close'][i-1]<=df['U-B'][i-1]:
                df.loc[ind[i],'U-B']=min(df['B-U'][i],df['U-B'][i-1])
            else: # Crossover
                df.loc[ind[i],'U-B']=df['B-U'][i]    
                
            if df['close'][i-1]>=df['L-B'][i-1]:
                df.loc[ind[i],'L-B']=max(df['B-L'][i],df['L-B'][i-1])
            else: # Crossover
                df.loc[ind[i],'L-B']=df['B-L'][i] 
    
    
        for test in range(window,len(df)):
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
    
    def search(self,multiplier_avg_candle_threshold=0.05,avg_candle_size_factor='median'):
        df = self.ohlc_df.copy(deep = True)
        
        avg_candle_size = at.compute_average_candle_size(avg_candle_size_factor,df)
        df["doji"] = abs(df["close"] - df["open"]) <=  (multiplier_avg_candle_threshold * avg_candle_size)
        return df
    
class Hammer(Security):
    
    def __init__(self,ohlc_df):
        Security.__init__(self,ohlc_df)
    
    def search(self,multiplier = 3,threshold = 0.6,err=0.001,percent_ratio = 0.1):   
        """returns dataframe with hammer candle column"""
        df = self.ohlc_df.copy(deep = True)
        df["hammer"] = (((df["high"] - df["low"])>multiplier*(df["open"] - df["close"])) & \
                       ((df["close"] - df["low"])/(err + df["high"] - df["low"]) > threshold) & \
                       ((df["open"] - df["low"])/(err+ df["high"] - df["low"]) > threshold)) & \
                       (abs(df["close"] - df["open"]) > percent_ratio* (df["high"] - df["low"]))
        return df

class ShootingStar(Security):
    
    def __init__(self,ohlc_df):
        Security.__init__(self,ohlc_df)
        
    def search(self,multiplier = 3,threshold = 0.6,err=0.001,percent_ratio = 0.1):
        df = self.ohlc_df.copy()
        df["sstar"] = (((df["high"] - df["low"])>multiplier*(df["open"] - df["close"])) & \
                   ((df["high"] - df["close"])/(err + df["high"] - df["low"]) > threshold) & \
                   ((df["high"] - df["open"])/(err + df["high"] - df["low"]) > threshold)) & \
                   (abs(df["close"] - df["open"]) > percent_ratio* (df["high"] - df["low"]))
        return df

class MaruBozu(Security):
    
    def __init__(self,ohlc_df):
        Security.__init__(self,ohlc_df)
        
    def search(self,multiplier_avg_candle=2,percent_ratio_avg_candle = 0.005,avg_candle_size_factor="median"):
        df = self.ohlc_df.copy()
        avg_candle_size = at.compute_average_candle_size(avg_candle_size_factor,df)
        df["h-c"] = df["high"]-df["close"]
        df["l-o"] = df["low"]-df["open"]
        df["h-o"] = df["high"]-df["open"]
        df["l-c"] = df["low"]-df["close"]
        df["maru_bozu"] = np.where((df["close"] - df["open"] > multiplier_avg_candle*avg_candle_size) & \
                                   (df[["h-c","l-o"]].max(axis=1) < percent_ratio_avg_candle*avg_candle_size),"maru_bozu_green",
                                   np.where((df["open"] - df["close"] > multiplier_avg_candle*avg_candle_size) & \
                                   (abs(df[["h-o","l-c"]]).max(axis=1) < percent_ratio_avg_candle*avg_candle_size),"maru_bozu_red",False))
        df.drop(["h-c","l-o","h-o","l-c"],axis=1,inplace=True)
        return df

############ Slope Trend #########################
class Slope(Security):
    
    def __init__(self,ohlc_df):
        Security.__init__(self,ohlc_df)
    
    def calculate(self,points):
        "function to calculate the slope of regression line for n consecutive points on a plot"
        df = self.ohlc_df.iloc[-1*points:,:]
        y = ((df["open"] + df["close"])/2).values
        x = np.array(range(points))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slope = np.rad2deg(np.arctan(results.params[-1]))
        return slope
    
class Trend(Security):
    
    def __init__(self,ohlc_df):
        Security.__init__(self,ohlc_df)
        

    
    def show(self,points,percent_ratio=0.7):
        "function to assess the trend by analyzing each candle"
        df = self.ohlc_df.copy()
        df["up"] = np.where(df["low"]>=df["low"].shift(1),1,0)
        df["dn"] = np.where(df["high"]<=df["high"].shift(1),1,0)
        if df["close"][-1] > df["open"][-1]:
            if df["up"][-1*points:].sum() >= percent_ratio*points:
                return "uptrend"
        elif df["open"][-1] > df["close"][-1]:
            if df["dn"][-1*points:].sum() >= percent_ratio*points:
                return "downtrend"
        else:
            return "None"
        #show(self.points,self.percent_ratio)