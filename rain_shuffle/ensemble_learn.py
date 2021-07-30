
import numpy as np
import os
from eval import evaluation, plot
from sklearn.externals import joblib
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from torch.autograd import Variable
import seaborn as sns

sns.set(style="white") #这是seaborn默认的风格
sns.set_palette("muted") #常用

'''
1. 必须把想要ensemble的模型保存在MODEL文件夹里，对于.pth模型必须要引入每个神经网络模型的类
2. 我修改了hour2txt.py文件，把数据集按照训练集和测试集分别保存在了./train_test_hour_shuffle文件夹以及./train_test_hour文件夹下
'''
from FNN import Net  
# 引入模型的类
# 。。。。
# 引入模型的类


def FakeBagging(path, dirlist, test_x, test_y):
    num = len(dirlist)
    y_pre = np.zeros((test_y.shape[0],1))
    rmse_list, mae_list, mdae_list, r2_list, var_list = [], [], [], [], []
    for ele in dirlist:
        a = ele.split('.')
        b = a[-1]
        if b=='m':
            clf = joblib.load('{}/{}'.format(path,ele))
            y_pre_temp = clf.predict(test_x)
            y_pre_temp = y_pre_temp.reshape(-1,1)
            y_pre += y_pre_temp
        else:
            if ele[:3]=='mlp':
                clf = torch.load('{}/{}'.format(path,ele),map_location=torch.device('cpu'))
                x_tensor = Tensor(test_x)
                x_tensor = Variable(x_tensor)
                # y_tensor = Tensor(test_y)
                y_pre_temp = clf(x_tensor).detach().numpy()
            else:
                clf = torch.load('{}/{}'.format(path,ele),map_location=torch.device('cpu'))
                x_tensor = Tensor(test_x)
                x_tensor = Variable(x_tensor).view(-1,7,3)
                # y_tensor = Tensor(test_y)
                y_pre_temp = clf(x_tensor).detach().numpy()
            y_pre += y_pre_temp
        rmse, mae, mdae, r2, var = evaluation(test_y, y_pre_temp)
#        print('model:',ele)
#        print('test_rmse: %r\n' % rmse,
#              'test_mae: %r\n' % mae,
#              'test_mdae: %r\n' % mdae,
#              'test_r2: %r\n' % r2,
#              'test_var: %r\n' % var)
        rmse_list.append(rmse); mae_list.append(mae); mdae_list.append(mdae); r2_list.append(r2); var_list.append(var)
    y_pre /= num
    rmse, mae, mdae, r2, var = evaluation(test_y, y_pre)
    rmse_list.append(rmse); mae_list.append(mae); mdae_list.append(mdae); r2_list.append(r2); var_list.append(var)
    
    # 绘图
    xmark = [ele.split('_')[0] for ele in dirlist]
    xmark.append('bagging_model')
    plt.figure();plt.plot(rmse_list,'o-',c='salmon', label = 'rmse');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.figure();plt.plot(mae_list,'*-',c='limegreen', label = 'mae');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.plot(mdae_list, 'v-',c='blue',label = 'mdae');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.plot(r2_list, '^-',c='cyan',label = 'r2_score');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.plot(var_list,'D-',c='darkorchid', label = 'var');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.show()
#    return rmse_list, mae_list, mdae_list, r2_list, var_list


def getStackingData(path, dirlist, train_x, train_y):
    num = len(dirlist)  # 基学习期个数
    n,m = train_x.shape

    data_x = np.zeros((n, num))  # 次级学习器的输入特征
    data_y = train_y             # 次级学习器的输出特征
    for i,ele in enumerate(dirlist):
        a = ele.split('.')
        b = a[-1]
        if b=='m':
            clf = joblib.load('{}/{}'.format(path,ele))
            y_pre_temp = clf.predict(train_x)
        else:
            if ele[:3] == 'mlp':
                clf = torch.load('{}/{}'.format(path, ele), map_location=torch.device('cpu'))
                x_tensor = Tensor(train_x)
                x_tensor = Variable(x_tensor)
                # y_tensor = Tensor(test_y)
                y_pre_temp = clf(x_tensor).detach().numpy()
            else:
                clf = torch.load('{}/{}'.format(path, ele), map_location=torch.device('cpu'))
                x_tensor = Tensor(train_x)
                x_tensor = Variable(x_tensor).view(-1, 7, 3)
                # y_tensor = Tensor(test_y)
                y_pre_temp = clf(x_tensor).detach().numpy()
        y_pre_temp = y_pre_temp.squeeze()
        data_x[:,i] = y_pre_temp
       
    return data_x, data_y


from sklearn.ensemble import GradientBoostingRegressor
def stackingUseGBRT(path, dirlist, est_g, dep_g, lr_g, train_x, train_y, test_x, test_y):
    num = len(dirlist)  # 基学习期个数
    rmse_list, mae_list, mdae_list, r2_list, var_list = [], [], [], [], []
    for ele in dirlist:
        a = ele.split('.')
        b = a[-1]
        if b=='m':
            clf = joblib.load('{}/{}'.format(path,ele))
            y_pre_temp = clf.predict(test_x)
        else:
            if ele[:3] == 'mlp':
                clf = torch.load('{}/{}'.format(path, ele), map_location=torch.device('cpu'))
                x_tensor = Tensor(test_x)
                x_tensor = Variable(x_tensor)
                # y_tensor = Tensor(test_y)
                y_pre_temp = clf(x_tensor).detach().numpy()
            else:
                clf = torch.load('{}/{}'.format(path, ele), map_location=torch.device('cpu'))
                x_tensor = Tensor(test_x)
                x_tensor = Variable(x_tensor).view(-1, 7, 3)
                # y_tensor = Tensor(test_y)
                y_pre_temp = clf(x_tensor).detach().numpy()
        y_pre_temp = y_pre_temp.reshape(-1,1)
        rmse, mae, mdae, r2, var = evaluation(test_y, y_pre_temp)
        rmse_list.append(rmse); mae_list.append(mae); mdae_list.append(mdae); r2_list.append(r2); var_list.append(var)
    
    data_x, data_y = getStackingData(path, dirlist, train_x, train_y)
    gbr = GradientBoostingRegressor(n_estimators=est_g, max_depth=dep_g, min_samples_split=3, learning_rate=lr_g)
    gbr.fit(data_x, data_y.ravel())
    test_x, test_y = getStackingData(path, dirlist, test_x, test_y)
    y_stacking = gbr.predict(test_x)
    y_stacking = y_stacking.reshape(-1,1)
    
    rmse, mae, mdae, r2, var = evaluation(test_y, y_stacking)
    rmse_list.append(rmse); mae_list.append(mae); mdae_list.append(mdae); r2_list.append(r2); var_list.append(var)
    # 绘图
    xmark = [ele.split('_')[0] for ele in dirlist]
    xmark.append('stacking_model')
    plt.figure();plt.plot(rmse_list, 'o-',c='salmon', label = 'rmse');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.figure();plt.plot(mae_list, '*-',c='limegreen', label = 'mae');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.plot(mdae_list,'v-',c='blue',label = 'mdae');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.plot(r2_list, '^-',c='cyan',label = 'r2_score');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.plot(var_list, 'D-',c='darkorchid', label = 'var');plt.legend();plt.xticks(range(num+1),xmark,rotation=45)
    plt.show()
#    return rmse_list, mae_list, mdae_list, r2_list, var_list



if __name__ == '__main__':
    pathnow = os.getcwd()
    path = '{}/MODEL'.format(pathnow)  
    dirlist = os.listdir(path)[1:]   #加载保存在MODEL文件夹里的所有.pth和.m模型
    
    # 加载测试数据集
    id_num = 371
    train_x=np.loadtxt("{}/train_test_hour_shuffle/train{}.txt".format(pathnow,id_num))
    train_y=np.loadtxt("{}/train_test_hour_shuffle/train_label{}.txt".format(pathnow,id_num))
    test_x=np.loadtxt("{}/train_test_hour_shuffle/test{}.txt".format(pathnow,id_num))
    test_y=np.loadtxt("{}/train_test_hour_shuffle/test_label{}.txt".format(pathnow,id_num))
    from sklearn.preprocessing import *
    train_x = scale(train_x,axis=0)
    train_y = np.reshape(train_y,(-1,1))
    test_x = scale(test_x,axis=0)
    test_y = np.reshape(test_y,(-1,1))
    
    
    # bagging
    FakeBagging(path, 
                dirlist, 
                test_x,
                test_y)
    
    # stacking use GBRT
    est_g = 90
    dep_g = 8
    lr_g = 0.02
    stackingUseGBRT(path, 
                    dirlist, 
                    est_g, 
                    dep_g, 
                    lr_g, 
                    train_x, 
                    train_y, 
                    test_x, 
                    test_y)

    







