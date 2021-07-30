import torch
import numpy as np
from sklearn.preprocessing import *
import os
from sklearn.model_selection import train_test_split
from eval import  evaluation
from LSTM_module import lstm
from sklearn.metrics import *
import matplotlib.pyplot as plt
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"]='2'
# torch.cuda.set_device(2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#设计传参数
parser = argparse.ArgumentParser()
parser.add_argument('--station', type=int, default=313, help='id of station')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=100, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--input-dim', type=int, default=3, help='num of hour')
parser.add_argument('--seq-len', type=int, default=7, help='num of parameter each hour')
parser.add_argument('--ifshuffle',action='store_true',help='shuffle data or not')
#超参数
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden unit number')
parser.add_argument('--n_layer', type=int, default=2, help='model layer number')

opt = parser.parse_args()

path =os.getcwd()
#station的id
num = opt.station

path_2 = 'train_test_hour'

if opt.ifshuffle:
      path_2 = 'train_test_hour_shuffle'

ratio=0.9
#和数据集设计有关
input_dim = opt.input_dim
seq_len= opt.seq_len
# 定义超参数
batch_size = opt.batch_size
learning_rate = opt.lr
num_epoches = opt.epochs
hidden_dim = opt.hidden_dim
n_layer = opt.n_layer


# 雨水信息
train_feature = np.loadtxt("{}/{}/train{}.txt".format(path,path_2, num)).astype(np.float32)
train_rain = np.loadtxt("{}/{}/train_label{}.txt".format(path,path_2, num)).astype(np.float32)
train_feature = scale(train_feature, axis=0)
train_rain = np.reshape(train_rain, (-1, 1))

test_feature = np.loadtxt("{}/{}/test{}.txt".format(path,path_2, num)).astype(np.float32)
test_rain = np.loadtxt("{}/{}/test_label{}.txt".format(path,path_2, num)).astype(np.float32)
test_feature = scale(test_feature, axis=0)
test_rain = np.reshape(test_rain, (-1, 1))



model = lstm(num,input_dim,seq_len,hidden_dim=hidden_dim,n_layer=n_layer,batch_size=batch_size,learning_rate=learning_rate,shuffle=False,device_pu=device)

eval_best=model.fit(train_feature,train_rain,num_epoches=num_epoches)
print("best val:"+str(eval_best)+'\n')

y_lstm=model.predict(test_feature,test_rain)

error = mean_squared_error(test_rain, y_lstm)
print('Model Test MSE: %.3f' % error)

rmse, mae, mdae,r2,var = evaluation(test_rain, y_lstm)

print('LSTM_rmse: %r' % rmse,
      'LSTM_mae: %r' % mae,
      'LSTM_mdae: %r' % mdae,
      'LSTM_r2: %r' % r2,
      'LSTM_var: %r' % var)


plt.plot(test_rain, 'b', label='real')
plt.plot(y_lstm, 'r', label='prediction',alpha=0.3)
plt.legend(loc='best')
plt.show()

#保存数据
path ='./LSTM'
f = open('{}/test_LSTM_{}.txt'.format(path,num), 'w+')
f.write('LSTM_rmse: %r ' % rmse +
      'LSTM_mae: %r ' % mae +
      'LSTM_mdae: %r ' % mdae +
      'LSTM_r2: %r ' % r2 +
      'LSTM_var: %r ' % var)
f.close()



