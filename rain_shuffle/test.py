import torch
import numpy as np
from sklearn.preprocessing import *
import os
from eval import  evaluation
from sklearn.metrics import *
import matplotlib.pyplot as plt
import argparse
from torch import Tensor
from torch.autograd import Variable

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# torch.cuda.set_device(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#设计传参数
parser = argparse.ArgumentParser()
parser.add_argument('--station', type=int, default=393, help='id of station')
parser.add_argument('--model_type',type=str,default='seq2seq')
parser.add_argument('--model',type=str,default='models/seq2seq_393_best.pth',help='model name')
parser.add_argument('--input-dim', type=int, default=3, help='num of hour')
parser.add_argument('--seq-len', type=int, default=7, help='num of parameter each hour')
parser.add_argument('--ifshuffle',action='store_true',help='shuffle data or not')
opt = parser.parse_args()

path =os.getcwd()
#station的id
num = opt.station

path_2 = 'sample'

if opt.ifshuffle:
      path_2 = 'sample_shuffle'

ratio=0.9
#和数据集设计有关
input_dim = opt.input_dim
seq_len= opt.seq_len

#测试集
test_feature = np.loadtxt("{}/{}/station{}/test{}.txt".format(path,path_2,num, num)).astype(np.float32)
test_rain = np.loadtxt("{}/{}/station{}/test_label{}.txt".format(path,path_2, num,num)).astype(np.float32)
test_feature = scale(test_feature, axis=0)
test_rain = np.reshape(test_rain, (-1, 1))


model = torch.load('{}/{}'.format(path,opt.model), map_location=torch.device('cpu'))
x_tensor = Tensor(test_feature)
if opt.model_type =='mlp':
      pass
else:
      x_tensor = Variable(x_tensor).view(-1, 7, 3)
y_pre_temp = model(x_tensor).detach().numpy()

error = mean_squared_error(test_rain, y_pre_temp)
print('Model Test MSE: %.3f' % error)

rmse, mae, mdae,r2,var = evaluation(test_rain, y_pre_temp)

print('SEQ2SEQ_rmse: %r' % rmse,
      'SEQ2SEQ_mae: %r' % mae,
      'SEQ2SEQ_mdae: %r' % mdae,
      'SEQ2SEQ_r2: %r' % r2,
      'SEQ2SEQ_var: %r' % var)

path ='./models'
fig = plt.figure()
plt.plot(test_rain, 'b', label='real')
plt.plot(y_pre_temp, 'r', label='prediction',alpha=0.3)
plt.legend(loc='best')
plt.show()
fig.savefig('{}/{}_{}.png'.format(path,opt.model_type,num), dpi=300)

f = open('{}/test_SEQ2SEQ_{}.txt'.format(path,num), 'w+')
f.write('SEQ2SEQ_rmse: %r ' % rmse +
      'SEQ2SEQ_mae: %r ' % mae +
      'SEQ2SEQ_mdae: %r ' % mdae +
      'SEQ2SEQ_r2: %r ' % r2 +
      'SEQ2SEQ_var: %r ' % var)
f.close()



