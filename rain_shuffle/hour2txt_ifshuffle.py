import pandas as pd
import numpy as np
import os
from random import shuffle
import sys

# 定义打乱函数
def myShuffle(X, y):
    y = y.reshape(-1,1)
    X_temp = np.concatenate((X, y),axis=1)
    shuffle(X_temp)
    return X_temp[:,:len(X_temp[0])-1], X_temp[:,-1].reshape(-1,1)

path = os.getcwd()
#特征的index
# feature_idx=[15,16,19,20,26,28,29]
feature_idx = [15,19,20,21,26,29,30]
rain_idx = [15]

time_ser = 3 #小时数
ratio = 0.9
ifshuffle = False

#该部分实现对全零数据的筛选
#k为计数有效数据
for num in [312,313,314,315,316,371,372,373,374,393,394,396]:
    k=0
    df = pd.read_csv('{}/hour_data/hour{}.csv'.format(path,num),engine='python')
    data_num = len(df)
    train_list=[]
    label_list=[]
    for i in range(data_num):
        j=i+1
        t = df.iloc[i, feature_idx].values
        if not np.any(t):
            k = 0
            continue
        else:
            k =k+1
        if j - time_ser >=0 and k == time_ser :
            train = df.iloc[j-time_ser:j, feature_idx].values
            label = df.iloc[j,rain_idx].values
            train=np.reshape(train,(1,-1))
            train_list.append(train)
            label_list.append(label)
            k =0
        if i >= data_num-2:
            break
    
    train_arr_old =np.array(train_list).reshape((len(train_list),-1))
    label_arr_old =np.array(label_list).reshape((len(label_list),-1))
    if ifshuffle:
        train_arr, label_arr = myShuffle(train_arr_old, label_arr_old)
        path ='./train_test_hour_shuffle'
    else:
        train_arr, label_arr = train_arr_old, label_arr_old
        path ='./train_test_hour'
    
    len_feature = len(train_arr)
    train_feature = train_arr[:int(ratio*len_feature),:]
    train_rain = label_arr[:int(ratio*len_feature),:]
    test_feature = train_arr[int(ratio*len_feature):,:]
    test_rain = label_arr[int(ratio*len_feature):,:]
    
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    
    np.savetxt('{}/train{}.txt'.format(path,num),train_feature)
    np.savetxt('{}/train_label{}.txt'.format(path,num),train_rain)
    np.savetxt('{}/test{}.txt'.format(path,num),test_feature)
    np.savetxt('{}/test_label{}.txt'.format(path,num),test_rain)
    
    path = os.getcwd()
    print('num%d结束'%num)








