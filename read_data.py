import pandas as pd
import scipy
from scipy import io
df = scipy.io.loadmat('df.mat')
feature = df["df"]
dfdata = pd.DataFrame(data=feature)
dfdata.to_csv('/Users/sun/Desktop/python-regression/data_tr.csv',index = True)
print (df.keys())

data = pd.read_csv('data_tr.csv', index_col = 0)
data.head()
print(data['0'])
