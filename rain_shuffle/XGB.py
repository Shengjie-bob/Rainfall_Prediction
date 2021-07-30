# 实现XGBoost回归, 以MSE损失函数为例
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import *
import xgboost as xgb
from xgboost import plot_importance
import joblib
from eval import evaluation
import argparse

if __name__ == '__main__':

    #设计传参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', type=int, default=371, help='id of station')
    parser.add_argument('--ifshuffle',action='store_true',help='shuffle data or not')
    opt = parser.parse_args()

    path =os.getcwd()
    #station的id
    num = opt.station

    path_2 = 'train_test_hour'

    if opt.ifshuffle:
        path_2 = 'train_test_hour_shuffle'

    ratio=0.9


    # 雨水信息
    train_feature = np.loadtxt("{}/{}/train{}.txt".format(path,path_2, num)).astype(np.float32)
    train_rain = np.loadtxt("{}/{}/train_label{}.txt".format(path,path_2, num)).astype(np.float32)
    train_feature = scale(train_feature, axis=0)
    train_rain = np.reshape(train_rain, (-1, 1))

    test_feature = np.loadtxt("{}/{}/test{}.txt".format(path,path_2, num)).astype(np.float32)
    test_rain = np.loadtxt("{}/{}/test_label{}.txt".format(path,path_2, num)).astype(np.float32)
    test_feature = scale(test_feature, axis=0)
    test_rain = np.reshape(test_rain, (-1, 1))


    x_train, x_test, y_train, y_test = train_test_split(train_feature, train_rain, test_size=0.22, random_state=42)

    """
    超参数
    """
    n_est = np.arange(10, 100, 10)
    n_dep = np.arange(1, 10, 1)
    n_lr = np.arange(0.01, 0.1, 0.01)
    paramters = ['lr', 'est', 'depth']

    path = './XGB'

    lr_g = 0.05
    est_g = 30
    dep_g = 3


    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    error = 100
    # f = open('{}/train_{}_{}.txt'.format(path,num, paramters[0]), 'w+')
    # for lr in n_lr:
    #     # 模型训练，使用XGB算法
    #     XGB = xgb.XGBRegressor(max_depth=dep_g, learning_rate=lr, n_estimators=est_g, silent=True, objective='reg:gamma')
    #     XGB.fit(x_train, y_train.ravel())
    #     y_pre_train = XGB.predict(x_train)
    #     y_pre_test = XGB.predict(x_test)
    #     error_1 = mean_squared_error(y_train, y_pre_train)
    #     print('Train MSE: %.3f' % error_1)
    #     error_2 = mean_squared_error(y_test, y_pre_test)
    #     print('Val MSE: %.3f' % error_2)

    #     f.write('{}: '.format(paramters[0]) + str(lr) +
    #             " Train_MSE: " + str(error_1) + ' Val_MSE: ' + str(error_2) + '\n')

    #     if error_2 < error:
    #         lr_g = lr
    #         error = error_2
    # f.close()


    # f = open('{}/train_{}_{}.txt'.format(path,num, paramters[1]), 'w+')
    # for est in n_est:
    #     # 模型训练，使用GBDT算法
    #     XGB = xgb.XGBRegressor(max_depth=dep_g, learning_rate=lr_g, n_estimators=est, silent=True, objective='reg:gamma')
    #     XGB.fit(x_train, y_train.ravel())
    #     y_pre_train = XGB.predict(x_train)
    #     y_pre_test = XGB.predict(x_test)
    #     error_1 = mean_squared_error(y_train, y_pre_train)
    #     print('Train MSE: %.3f' % error_1)
    #     error_2 = mean_squared_error(y_test, y_pre_test)
    #     print('Val MSE: %.3f' % error_2)

    #     f.write('{}: '.format(paramters[1]) + str(est) +
    #             " Train_MSE: " + str(error_1) + ' Val_MSE: ' + str(error_2) + '\n')

    #     if error_2 < error:
    #         est_g = est
    #         error = error_2
    # f.close()


    # f = open('{}/train_{}_{}.txt'.format(path,num, paramters[2]), 'w+')
    # for dep in n_dep:
    #     # 模型训练，使用GBDT算法
    #     XGB = xgb.XGBRegressor(max_depth=dep, learning_rate=lr_g, n_estimators=est_g, silent=True, objective='reg:gamma')
    #     XGB.fit(x_train, y_train.ravel())
    #     y_pre_train = np.nan_to_num(XGB.predict(x_train))
    #     y_pre_test = np.nan_to_num(XGB.predict(x_test))
    #     error_1 = mean_squared_error(y_train, y_pre_train)
    #     print('Train MSE: %.3f' % error_1)
    #     error_2 = mean_squared_error(y_test, y_pre_test)
    #     print('Val MSE: %.3f' % error_2)

    #     f.write('{}: '.format(paramters[2]) + str(dep) +
    #             " Train_MSE: " + str(error_1) + ' Val_MSE: ' + str(error_2) + '\n')

    #     if error_2 < error:
    #         dep_g = dep
    #         error = error_2
    # f.close()

    """
    保存模型 加载模型 测试算法
    """
    # 保存模型 加载模型 测试算法
    XGB = xgb.XGBRegressor(max_depth=3, learning_rate=0.03, n_estimators=80, silent=True, objective='reg:gamma')
    XGB.fit(x_train, y_train.ravel())
    joblib.dump(XGB, '{}/train_XGB_{}.m'.format(path,num))  # 保存模型
    clf = joblib.load('{}/train_XGB_{}.m'.format(path,num))
    y_xgb = clf.predict(test_feature)
    error = mean_squared_error(test_rain, y_xgb)
    print('Model Test MSE: %.3f' % error)

    plt.plot(test_rain, 'b', label='real')
    plt.plot(y_xgb, 'r', label='prediction', alpha=0.3)
    plt.legend(loc='best')
    plt.show()

    plot_importance(XGB)
    plt.show()

    rmse, mae, mdae, r2, var = evaluation(test_rain, y_xgb)

    print('XGB_rmse: %r' % rmse,
          'XGB_mae: %r' % mae,
          'XGB_mdae: %r' % mdae,
          'XGB_r2: %r' % r2,
          'XGB_var: %r' % var)

    f = open('{}/test_XGB_{}.txt'.format(path,num), 'w+')
    f.write('XGB_rmse: %r ' % rmse +
      'XGB_mae: %r ' % mae +
      'XGB_mdae: %r ' % mdae +
      'XGB_r2: %r ' % r2 +
      'XGB_var: %r ' % var)
    f.close()
