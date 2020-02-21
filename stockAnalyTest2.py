import pandas as pd
import numpy as np
import plotly.offline as py
import plotly.graph_objects as go
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import seaborn as sns

def excel_to_df(excel_sheet):
	df = pd.read_excel(excel_sheet)
	df.dropna(how='all', inplace=True)

	index_PL = int(df.loc[df['Data provided by SimFin']=='Profit & Loss statement'].index[0])
	index_CF = int(df.loc[df['Data provided by SimFin']=='Cash Flow statement'].index[0])
	index_BS = int(df.loc[df['Data provided by SimFin']=='Balance Sheet'].index[0])

	df_PL = df.iloc[index_PL:index_BS-1, 1:]
	df_PL.dropna(how='all', inplace=True)
	df_PL.columns = df_PL.iloc[0]
	df_PL = df_PL[1:]
	df_PL.set_index("in million USD", inplace=True)
	(df_PL.fillna(0, inplace=True))
	

	df_BS = df.iloc[index_BS-1:index_CF-2, 1:]
	df_BS.dropna(how='all', inplace=True)
	df_BS.columns = df_BS.iloc[0]
	df_BS = df_BS[1:]
	df_BS.set_index("in million USD", inplace=True)
	df_BS.fillna(0, inplace=True)
	

	df_CF = df.iloc[index_CF-2:, 1:]
	df_CF.dropna(how='all', inplace=True)
	df_CF.columns = df_CF.iloc[0]
	df_CF = df_CF[1:]
	df_CF.set_index("in million USD", inplace=True)
	df_CF.fillna(0, inplace=True)
	
	df_CF = df_CF.T
	df_BS = df_BS.T
	df_PL = df_PL.T
    
	return df, df_PL, df_BS, df_CF

def combine_regexes(regexes):
	return "(" + ")|(".join(regexes) + ")"

df,df_PL,df_BS,df_CF=excel_to_df("C:/PythonAI/SimFin-data.xlsx")
df_BS=df_BS.iloc[:,1:]

assets = go.Bar(
    x=df_BS.index,
    y=df_BS["Total Assets"],
    name='Assets'
)
liabilities = go.Bar(
    x=df_BS.index,
    y=df_BS["Total Liabilities"],
    name='Liabilities'
)

shareholder_equity = go.Scatter(
    x=df_BS.index,
    y=df_BS["Total Equity"],
    name='Equity'
)

data = [assets, liabilities, shareholder_equity]
layout = go.Layout(
    barmode='stack'
)

fig_bs = go.Figure(data=data, layout=layout)
py.plot(fig_bs, filename='Total Assets and Liabilities')


asset_data = []
columns = '''
Cash, Cash Equivalents & Short Term Investments
Accounts & Notes Receivable
Inventories
Other Short Term Assets
Property, Plant & Equipment, Net
Long Term Investments & Receivables
Other Long Term Assets
'''


for col in columns.strip().split("\n"):
    asset_bar = go.Bar(
        x=df_BS.index,
        y=df_BS[ col ],
        name=col
    )    
    asset_data.append(asset_bar)
    
layout_assets = go.Layout(
    barmode='stack'
)

fig_bs_assets = go.Figure(data=asset_data, layout=layout_assets)
py.plot(fig_bs_assets, filename='Total Assets Breakdown')

liability_data = []
columns = '''
Payables & Accruals
Short Term Debt
Other Short Term Liabilities
Long Term Debt
Other Long Term Liabilities
'''


for col in columns.strip().split("\n"):
    liability_bar = go.Bar(
        x=df_BS.index,
        y=df_BS[ col ],
        name=col
    )    
    liability_data.append(liability_bar)
    
layout_liabilitys = go.Layout(
    barmode='stack'
)

fig_bs_liabilitys = go.Figure(data=liability_data, layout=layout_liabilitys)
py.plot(fig_bs_liabilitys, filename='Total liabilitys Breakdown')


df_BS["working capital"] = df_BS["Total Current Assets"] - df_BS["Total Current Liabilities"]
df_BS["working capital"].plot()


data4Predict = df_BS.iloc[:,0:0]

predict = "알고싶은 칼럼 입력."

X = np.array(df_BS.drop([predict], 1))
X2 = np.array(data4Predict.drop([predict], 1))
y = np.array(df_BS[predict])
y2 = np.array(data4Predict[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

#predictions = linear.predict(x_test)
predictions = linear.predict(X2)
print("predictions: \n",predictions)

#for x in range(len(predictions)):
#    print("result: \n",predictions[x],", ", x_test[x], y_test[x])

plt.plot(X2,predictions)
plt.xlabel('input X')
plt.ylabel('outcome Y')
plt.title('Experiment Result')
plt.show()

sns.heatmap(data.corr())