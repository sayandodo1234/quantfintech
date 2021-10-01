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
from .Security_init import Security

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

        
    
       
                
