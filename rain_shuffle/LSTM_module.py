import torch
from torch import nn, optim
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import *
import os
from sklearn.model_selection import train_test_split
from eval import evaluation
from sklearn.metrics import *

path = './LSTM'


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


# 定义 Recurrent Network 模型
class LSTM_module(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(LSTM_module, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x, None)

        out = self.classifier(out[:, -1, :])
        out = torch.relu(out)
        return out


class lstm():
    def __init__(self, num, input_dim, seq_len, hidden_dim, n_layer, batch_size=100, learning_rate=1e-3, shuffle=True,
                 device_pu='cpu'):
        self.station = num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.seq_len = seq_len
        self.bs = batch_size
        self.lr = learning_rate
        self.shuffle = shuffle
        self.device = device_pu
        self.model = LSTM_module(input_dim, hidden_dim, n_layer, 1).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, data, label, num_epoches=100):

        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.22, random_state=42)

        train = GetLoader(x_train, y_train)
        data_train = torch.utils.data.DataLoader(train, batch_size=self.bs, shuffle=self.shuffle)
        test = GetLoader(x_test, y_test)
        data_test = torch.utils.data.DataLoader(test, batch_size=self.bs, shuffle=self.shuffle)

        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)

        eval_loss_best = np.inf

        uncorrect =True
        while uncorrect:
            self.model = LSTM_module(self.input_dim, self.hidden_dim, self.n_layer, 1).to(self.device)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            f = open('{}/train_{}_{}_{}_{}.txt'.format(path, self.station, self.hidden_dim, self.n_layer, self.lr), 'w+')
            train_loss_last = np.inf
            # 开始训练
            for epoch in range(num_epoches):
                self.model.train()
                print('epoch {}'.format(epoch + 1))
                print('**************************************')
                running_loss = 0.0

                for i, data in enumerate(data_train, 1):
                    """
                    随机打乱的方式不好 应该是全部打乱之后 固定抽取 否则会出现样本利用不均衡的问题
                    """
                    img, label = data
                    img = Variable(img).to(self.device)
                    label = Variable(label).to(self.device)
                    # 向前传播
                    out = self.model(img.view(-1, self.seq_len, self.input_dim))
                    loss = self.criterion(out, label)
                    running_loss += loss.data.item() * label.size(0)
                    # 向后传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_loss = running_loss / (len(y_train))
                print('Finish {} epoch, Loss: {:.6f}'.format(
                    epoch + 1, train_loss))

                if train_loss_last == train_loss  and epoch < 3:
                    break
                if train_loss_last > train_loss and epoch >=3:
                    uncorrect = False
                
                train_loss_last = train_loss

                self.model.eval()
                eval_loss = 0.
                for data in data_test:
                    img, label = data

                    img = Variable(img).to(self.device)
                    label = Variable(label).to(self.device)
                    out = self.model(img.view(-1, self.seq_len, self.input_dim))
                    loss = self.criterion(out, label)
                    eval_loss += loss.data.item() * label.size(0)
                val_loss = eval_loss / (len(y_test))
                print('Val Loss: {:.6f}'.format(val_loss))

                f.write(" Train_MSE: " + str(train_loss) + ' Val_MSE: ' + str(val_loss) + '\n')

                if val_loss < eval_loss_best:
                    eval_loss_best = val_loss
                    self.eval = eval_loss_best
                    torch.save(self.model,
                            '{}/lstm_{}_{}_{}_{}.pth'.format(path, self.station, self.hidden_dim, self.n_layer, self.lr))
            f.close()

        return self.eval

    def predict(self, test_data, test_label):

        test_model = torch.load(
            '{}/lstm_{}_{}_{}_{}.pth'.format(path, self.station, self.hidden_dim, self.n_layer, self.lr)).to(
            self.device)

        test_loss = 0
        criterion = nn.MSELoss()
        test = GetLoader(test_data, test_label)
        data_test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

        y_mlp = []
        for data in data_test:
            img, label = data
            img = Variable(img).to(self.device)
            label = Variable(label).to(self.device)
            out = test_model(img.view(-1, self.seq_len, self.input_dim))
            loss = criterion(out, label)
            test_loss += loss.data
            y_mlp.append(out.data)

        print('Test Loss: {:.6f}'.format(test_loss / (len(
            test_label))))

        y_mlp = np.array(y_mlp).squeeze()[:, np.newaxis]

        return y_mlp








