import torch
import numpy as np
from sklearn.preprocessing import *
import os
from sklearn.model_selection import train_test_split
from eval import  evaluation
from seq2seq_module import Seq2Seq
from sklearn.metrics import *
import matplotlib.pyplot as plt
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# torch.cuda.set_device(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#设计传参数
parser = argparse.ArgumentParser()
parser.add_argument('--station', type=int, default=371, help='id of station')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=30, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--input-dim', type=int, default=3, help='num of hour')
parser.add_argument('--seq-len', type=int, default=7, help='num of parameter each hour')
parser.add_argument('--ifshuffle',action='store_true',help='shuffle data or not')
#超参数
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden unit number')

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


# 雨水信息
train_feature = np.loadtxt("{}/{}/train{}.txt".format(path,path_2, num)).astype(np.float32)
train_rain = np.loadtxt("{}/{}/train_label{}.txt".format(path,path_2, num)).astype(np.float32)
train_feature = scale(train_feature, axis=0)
train_rain = np.reshape(train_rain, (-1, 1))

test_feature = np.loadtxt("{}/{}/test{}.txt".format(path,path_2, num)).astype(np.float32)
test_rain = np.loadtxt("{}/{}/test_label{}.txt".format(path,path_2, num)).astype(np.float32)
test_feature = scale(test_feature, axis=0)
test_rain = np.reshape(test_rain, (-1, 1))



model = Seq2Seq(num,input_dim,seq_len,output_dim=1,hidden_size=hidden_dim,dropout=0.1,learning_rate=learning_rate,batch_size=batch_size,device_pu=device)

eval_best=model.fit(train_feature,train_rain,num_epoches=num_epoches,shuffle=False)
print("best val:"+str(eval_best)+'\n')

y_seq2seq=model.predict(test_feature,test_rain)

error = mean_squared_error(test_rain, y_seq2seq)
print('Model Test MSE: %.3f' % error)

rmse, mae, mdae,r2,var = evaluation(test_rain, y_seq2seq)

print('SEQ2SEQ_rmse: %r' % rmse,
      'SEQ2SEQ_mae: %r' % mae,
      'SEQ2SEQ_mdae: %r' % mdae,
      'SEQ2SEQ_r2: %r' % r2,
      'SEQ2SEQ_var: %r' % var)


plt.plot(test_rain, 'b', label='real')
plt.plot(y_seq2seq, 'r', label='prediction',alpha=0.3)
plt.legend(loc='best')
plt.show()


path ='./SEQ2SEQ'
f = open('{}/test_SEQ2SEQ_{}.txt'.format(path,num), 'w+')
f.write('SEQ2SEQ_rmse: %r ' % rmse +
      'SEQ2SEQ_mae: %r ' % mae +
      'SEQ2SEQ_mdae: %r ' % mdae +
      'SEQ2SEQ_r2: %r ' % r2 +
      'SEQ2SEQ_var: %r ' % var)
f.close()



