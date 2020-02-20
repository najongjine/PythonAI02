# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:54:54 2020

@author: 505-06
"""

import pandas_datareader as web
df = web.DataReader('apple', data_source='yahoo', start='2012-01-01', end='2020-2-14')
print(df)