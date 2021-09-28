#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:12:32 2021

@author: sayandebhowmick
"""
import sys
import pandas as pd

def exponential_moving_parameter_check(decay_type,alpha_val):
    #if exponential_moving == False:
        #return False
        #if decay_type != None:
            #sys.exit("Cannot have decay type when exponential moving is false")
        #if (alpha>=0) and (alpha<=1):
            #sys.exit("Cannot have alpha value when exponential moving is false")
    #else:
    if (alpha_val<0) or (alpha_val>1):
        if decay_type == None:
            sys.exit("Either specify decay type or alpha value")
        else:
            if (decay_type != "com") and (decay_type != "span") and (decay_type != "halflife"):
                sys.exit("Not a valid decay type")
    else:
        if decay_type != None:
            print("Parameter alpha is given more preference over decay type")
    return 


def compute_exponential_moving(ser,window,decay_type,decay_value,alpha_val):
    def determine_decay_value(decay_value,window):
        if decay_value == -1.0:
            mod_decay_value = window
        else:
            mod_decay_value = decay_value
        return mod_decay_value
    
    if alpha_val>=0 and alpha_val<=1:
        ans_ser = ser.ewm(alpha=alpha_val,min_periods=window)
    else:
        final_decay_value = determine_decay_value(decay_value,window)
        if decay_type == "com":
            ans_ser = ser.ewm(com=final_decay_value, min_periods=window)
        if decay_type == "span":
            ans_ser = ser.ewm(span=final_decay_value, min_periods=window)
        if decay_type == "halflife":
            ans_ser = ser.ewm(halflife=final_decay_value, min_periods=window)
    
    return ans_ser

def compute_average_candle_size(avg_candle_size_factor,df):
    if avg_candle_size_factor == 'mean':
        avg_candle_size = abs(df["close"] - df["open"]).mean()
    elif avg_candle_size_factor == 'median':
        avg_candle_size = abs(df["close"] - df["open"]).median()
    else:
        sys.exit("Not a valid candle size factor")
    return avg_candle_size 