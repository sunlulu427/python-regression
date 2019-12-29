import pandas as pd
import scipy
from scipy import io
import numpy
import time

# 读取.mat文件,并将文件转化为.csv文件
df = scipy.io.loadmat('beta_estimate.mat')

feature = df["beta"]

dfdata = pd.DataFrame(data=feature)

dfdata.to_csv('/Users/sun/Desktop/python-regression/data_tr.csv', index = True)


data = pd.read_csv('data_tr.csv', index_col = 0)
data.head()

t = numpy.arange(0, len(data) / 100.0, 0.01)
print('数据的长度:', len(t))
print('时间：', t)
vals = data['0'].values
print('数据', vals)
