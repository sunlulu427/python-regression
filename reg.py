import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import io
# 正常显示中文
import matplotlib as mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示符号
from matplotlib import rcParams
rcParams['axes.unicode_minus'] = False

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 载入模型库
# 1.线性回归
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
# 2.决策树回归
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
# 3.支持向量机回归
from sklearn import svm
model_SVR = svm.SVR()
# 4.K近邻回归
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
# 5.随机森林回归
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators = 20)
# 6.AdaBoost回归
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators = 50)
# 7.梯度增强随机森林回归
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators = 100)
# 8.bagging 回归
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
# 9.ExtraTree回归
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()


# 读取.mat文件,并将文件转化为.csv文件
df = scipy.io.loadmat('beta_estimate.mat')
feature = df["beta"]
dfdata = pd.DataFrame(data=feature)
dfdata.to_csv('/Users/sun/Desktop/python-regression/data_tr.csv', index = True)
data = pd.read_csv('data_tr.csv', index_col = 0)
data.head()
t = np.arange(0, len(data) / 100.0, 0.01).reshape(-1, 1)
print('数据的长度:', len(t))
print('时间：', t)

time = np.arange(0, 1.5 * len(data) / 100, 0.01).reshape(-1, 1)

vals = data['0'].values
print('数据', vals)

# 使用一元线性回归检验数据集
#model_LinearRegression.fit(t, vals)
#y_pred = model_LinearRegression.predict(t);
#plt.figure(figsize=(14, 4))
#plt.scatter(t, vals, color='g')
#plt.plot(t, y_pred, color='r')
#plt.xlabel('time(0-25)')
#plt.show()

# 定义函数
def model_fit(model):
	model.fit(t, vals)
	y_pred = model.predict(t)
	plt.figure(figsize=(14,4))
	plt.plot(t, vals, color='g')
	plt.plot(t, y_pred, color='r')
	plt.show()
	print('Score:+++', model.score(t, vals))


def model_pre(regr):
	model.fit(t, vals)
	y_pred = model.predict(time)
	plt.figure(figsize=(14,4))
	plt.plot(t, vals, color='g')
	plt.plot(time, y_pred, color='r')
	plt.show()
	print('Score:+++', model.score(t, vals))

model_fit(model_DecisionTreeRegressor)

# 定义多元模型
def model_function(regr):
	pf = PolynomialFeatures(degree=2)
	regr.fit(pf.fit_transform(t), vals)
	x_predict = np.linspace(0, 2500, 1440)
	x_predict_result = regr.predict(pf.transform(x_predict.reshape(x_predict.shape[0], 1)))
	plt.figure(figsize=(14,4))
	plt.scatter(t, vals, color='g')
	plt.plot(x_predict,x_predict_result, 'r-')
	plt.xlabel('time')
	plt.show()



