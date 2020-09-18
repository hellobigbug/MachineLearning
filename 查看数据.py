#!/usr/bin/env python 
# encoding: utf-8 
"""
 @Author : hanxiaopeng
 @Time : 2020/9/17 
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("data.csv", header=0)
df = df.set_index('date')
df = df['2017/4/1':'2018/6/1']
# df = df.drop(index='2017/6/11')
# df = df.drop(index='2017/6/12')
# df = df.drop(index='2017/6/13')
# df = df.drop(index='2017/6/23')
# df = df.drop(index='2017/9/11')
# df = df.drop(index='2017/8/26')
# df = df.drop(index='2017/8/27')
df.to_csv("data.csv")
column = df.columns
print(len(column))
for i in column:
    df1 = df[i]
    df1.plot()
    plt.show()
