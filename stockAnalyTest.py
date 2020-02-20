# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:12:48 2020

@author: 505-06
"""
import seaborn as sns
import pandas as pd
df=pd.read_excel("SimFin-data.xlsx")

#df['data pri...'] 열에서 'profit & ...' 행을 찾고 0번째 원소의 행 번호.
index_PL = df.loc[ df["Data provided by SimFin"] == 'Profit & Loss statement'].index[0]

print(index_PL)

index_CF = int(df.loc[df['Data provided by SimFin']=='Cash Flow statement'].index[0])
index_BS = int(df.loc[df['Data provided by SimFin']=='Balance Sheet'].index[0])

print(index_CF)
print(index_BS)

#index_pl+1 행번째부터 index_bs-1의 행까지, 1 index째 칼럼부터 전체 칼럼.
df_PL = df.iloc[index_PL+1:index_BS-1, 1:]
print(df_PL)

# 칼럼의 값을 0번째 행의 값에서 복사해서 세팅.
df_PL.columns = df_PL.iloc[0]
df_PL.set_index("in million USD",inplace=True)

########## 내가 작성한거.
df_PL = df_PL.iloc[1:,:]
df_PL=df_PL.T

# NaN 을 0로 변경.
df_PL.fillna(0, inplace=True)

############## 내가 작성한
sns.heatmap(df_PL.corr())