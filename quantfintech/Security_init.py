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