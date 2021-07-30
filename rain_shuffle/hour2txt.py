import pandas as pd
import numpy as np
import os

path = os.getcwd()

#选择数据集
num=396

df = pd.read_csv('{}/hour_data/hour{}.csv'.format(path,num))

data_num = len(df)
#特征的index
# feature_idx=[15,16,19,20,26,28,29]
feature_idx=[15,19,20,21,26,29,30]
rain_idx =[15]
#天数
time_ser=3

#数据集划分
ratio = 0.9

train_list=[]
label_list=[]

#该部分实现对全零数据的筛选
#k为计数有效数据
k=0
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

train_arr =np.array(train_list).reshape((len(train_list),-1))
label_arr =np.array(label_list).reshape((len(label_list),-1))

path ='./train_test_hour'
if os.path.exists(path):
    pass
else:
    os.mkdir(path)

np.savetxt('{}/train{}.txt'.format(path,num),train_arr)
np.savetxt('{}/label{}.txt'.format(path,num),label_arr)

print('ss')








