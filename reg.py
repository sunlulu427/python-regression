import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 正常显示中文
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 正常显示符号
from matplotlib import rcParams
rcParams['axes.unicode_minus'] = False

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PloynomialFeatures

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
