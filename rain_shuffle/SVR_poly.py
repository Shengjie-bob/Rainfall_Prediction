import pandas as pd
from sklearn.preprocessing import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import os
from sklearn.model_selection import train_test_split
import joblib
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


feature = np.loadtxt("{}/{}/train{}.txt".format(path,path_2, num))
rain = np.loadtxt("{}/{}/label{}.txt".format(path, path_2,num))

feature = scale(feature, axis=0)
rain = np.reshape(rain, (-1, 1))
# min_max_scaler = MinMaxScaler()
# rain = min_max_scaler.fit_transform(rain)

# 雨水信息
train_feature = feature[:int(ratio * len(feature)), :]
train_rain = rain[:int(ratio * len(feature)), :]

test_feature = feature[int(ratio * len(feature)):, :]
test_rain = rain[int(ratio * len(feature)):, :]

x_train, x_test, y_train, y_test = train_test_split(train_feature, train_rain, test_size=0.22, random_state=42)


"""
超参数
"""
n_gamma = np.arange(0.01,0.1,0.01)
n_C =np.arange(1,100,10)
n_degree =np.arange(1,10,1)
paramters =['C','gamma','degree']
kernel ='poly'

path ='./SVR_{}'.format(kernel)


gamma_g=0.01
c_g=50
degree_g=3

if os.path.exists(path):
    pass
else:
    os.mkdir(path)


error = 100
f = open('{}/train_{}_{}.txt'.format(path,num,paramters[0]), 'w+')
for c in n_C:
    # 模型训练
    regressor = SVR(kernel = kernel,gamma=gamma_g,C=c,degree=degree_g)
    regressor.fit(x_train, y_train.ravel())
    y_pre_train = regressor.predict(x_train)
    y_pre_test = regressor.predict(x_test)
    error_1 = mean_squared_error(y_train, y_pre_train)
    print('Train MSE: %.3f' % error_1)
    error_2 = mean_squared_error(y_test, y_pre_test)
    print('Val MSE: %.3f' % error_2)

    f.write('{}: '.format(paramters[0]) + str(c) +
            " Train_MSE: " + str(error_1) +' Val_MSE: '+ str(error_2)+'\n')

    if error_2 < error:
        c_g = c
        error =error_2
f.close()


f = open('{}/train_{}_{}.txt'.format(path,num,paramters[1]), 'w+')
for gamma in n_gamma:
    # 模型训练，使用GBDT算法
    regressor = SVR(kernel = kernel,gamma=gamma,C=c_g,degree=degree_g)
    regressor.fit(x_train, y_train.ravel())
    y_pre_train = regressor.predict(x_train)
    y_pre_test = regressor.predict(x_test)
    error_1 = mean_squared_error(y_train, y_pre_train)
    print('Train MSE: %.3f' % error_1)
    error_2 = mean_squared_error(y_test, y_pre_test)
    print('Val MSE: %.3f' % error_2)

    f.write('{}: '.format(paramters[1]) + str(gamma) +
            " Train_MSE: " + str(error_1) +' Val_MSE: '+ str(error_2)+'\n')
    if error_2 <error:
        gamma_g = gamma
        error = error_2
f.close()

f = open('{}/train_{}_{}.txt'.format(path,num,paramters[2]), 'w+')
for degree in n_degree:
    # 模型训练，使用GBDT算法
    regressor = SVR(kernel = kernel,gamma=gamma_g,C=c_g,degree=degree)
    regressor.fit(x_train, y_train.ravel())
    y_pre_train = regressor.predict(x_train)
    y_pre_test = regressor.predict(x_test)
    error_1 = mean_squared_error(y_train, y_pre_train)
    print('Train MSE: %.3f' % error_1)
    error_2 = mean_squared_error(y_test, y_pre_test)
    print('Val MSE: %.3f' % error_2)

    f.write('{}: '.format(paramters[2]) + str(degree) +
            " Train_MSE: " + str(error_1) +' Val_MSE: '+ str(error_2)+'\n')

    if error_2 <error:
        degree_g = degree
        error = error_2
f.close()


"""
保存模型 加载模型 测试算法
"""
#保存模型 加载模型 测试算法
regressor = SVR(kernel=kernel, gamma=gamma_g, C=c_g,degree=degree_g)
regressor.fit(x_train, y_train.ravel())
joblib.dump(regressor, '{}/train_{}_{}.m'.format(path,kernel,num))   # 保存模型
clf=joblib.load('{}/train_{}_{}.m'.format(path,kernel,num))
y_svr = clf.predict(test_feature)
error = mean_squared_error(test_rain, y_svr)
print('Model Test MSE: %.3f' % error)

plt.plot(test_rain, 'b', label='real')
plt.plot(y_svr, 'r', label='prediction',alpha=0.3)
plt.legend(loc='best')
plt.show()

rmse, mae, mdae,r2,var = evaluation(test_rain, y_svr)

print('SVR_rmse: %r' % rmse,
      'SVR_mae: %r' % mae,
      'SVR_mdae: %r' % mdae,
      'SVR_r2: %r' % r2,
      'SVR_var: %r' % var)

f = open('{}/test_SVR_{}_{}.txt'.format(path,kernel,num), 'w+')
f.write('SVR_rmse: %r ' % rmse +
      'SVR_mae: %r ' % mae +
      'SVR_mdae: %r ' % mdae +
      'SVR_r2: %r ' % r2 +
      'SVR_var: %r ' % var)
f.close()