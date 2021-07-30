import pandas as pd
from sklearn.preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from pandas.plotting import *
from statsmodels.graphics.tsaplots import *
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import os
from eval import evaluation
import argparse

#设计传参数
parser = argparse.ArgumentParser()
parser.add_argument('--station', type=int, default=371, help='id of station')
opt = parser.parse_args()

path =os.getcwd()
#station的id
num = opt.station
path_2 ='train_test_hour'
ratio=0.9


rain=np.loadtxt("{}/{}/label{}.txt".format(path,path_2,num))
# min_max_scaler = MinMaxScaler()
# rain = min_max_scaler.fit_transform(rain)

# df = pd.read_csv('tian.csv')
#
# rain_series=df.iloc[:4000,16]

autocorrelation_plot(rain)
plt.show()
#
# lag_plot(rain_series)
# plt.show()

# plot_acf(rain_series)
# plt.show()
#
# diff1 = rain_series.diff(1).dropna()
# diff1.plot()
# plt.show()


# plot_acf(diff1)
# plt.show()
#
# plot_pacf(rain_series)
# plt.show()

# print(u'差分序列的白噪声检验结果为：', acorr_ljungbox( diff1, lags=1))
#
# model = ARIMA(diff1[:1000], (0,1,2)).fit()
# model.summary2()
# re=model.fittedvalues
# results=model.predict()
#
#
# output = model.forecast()


X = rain
size = int(len(X) * ratio)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history[:5000], order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)



rmse, mae, mdae,r2,var = evaluation(test, predictions)

print('ARIMA_rmse: %r' % rmse,
      'ARIMA_mae: %r' % mae,
      'ARIMA_mdae: %r' % mdae,
      'ARIMA_r2: %r' % r2,
      'ARIMA_var: %r' % var)


plt.plot(test, 'b', label='real')
plt.plot(predictions, 'r', label='prediction',alpha=0.3)
plt.legend(loc='best')
plt.show()

"""
保存数据
"""

path ='./ARIMA'
"""
建立文件夹
"""
if os.path.exists(path):
    pass
else:
    os.mkdir(path)

f = open('{}/test_ARIMA_{}.txt'.format(path,num), 'w+')
f.write('ARIMA_rmse: %r ' % rmse +
      'ARIMA_mae: %r ' % mae +
      'ARIMA_mdae: %r ' % mdae +
      'ARIMA_r2: %r ' % r2 +
      'ARIMA_var: %r ' % var)
f.close()
