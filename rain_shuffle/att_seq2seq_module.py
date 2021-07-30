from __future__ import unicode_literals, print_function, division
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import numpy as np



MAX_LENGTH=10
path ='./ATT_SEQ2SEQ'


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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size,batch_first=True)

    def forward(self, input, hidden):
        # embedded = self.embedding(input).view(-1, 1 ,self.hidden_size)
        # output = embedded
        output = input.view(-1, 1, self.input_size)
        output, hidden = self.gru(output, hidden)
        output = torch.relu(output)
        return output, hidden

    def initHidden(self,batch_size):
        return torch.zeros(1, batch_size , self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size,dropout_p,seq_len):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.embedding = nn.Linear(hidden_size, hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.seq_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)


    def forward(self, input, hidden,encoder_outputs):
        # output = self.embedding(input).view(-1, 1 ,self.hidden_size)

        output = input.view(1, -1, self.hidden_size)

        attn_weights = F.softmax(
            self.attn(torch.cat((output[0], hidden[0]), 1)), dim=1)

        attn_weights=attn_weights.unsqueeze(0)
        attn_weights = attn_weights.transpose(1,0)

        encoder_outputs = encoder_outputs.transpose(1, 0)
        attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs).transpose(1,0)

        output = torch.cat((output[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output).transpose(1,0)
        output, hidden = self.gru(output, hidden)

        output = torch.relu(self.out(output[:,0,:]))

        return output, hidden,attn_weights

    def initHidden(self,batch_size):
        return torch.zeros(1,batch_size, self.hidden_size)



class seq2seq_cell(nn.Module):
    def __init__(self,input_dim,seq_len,output_dim,hidden_size,dropout,learning_rate,batch_size,device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr = learning_rate
        self.bs = batch_size
        self.device = device
        self.encoder = EncoderRNN(self.input_dim, self.hidden_size)
        self.decoder = DecoderRNN(self.hidden_size, self.output_dim, dropout_p=self.dropout,seq_len=self.seq_len)
    def forward(self,input):

        encoder_hidden = self.encoder.initHidden(input.shape[0])
        encoder_outputs = torch.zeros( self.seq_len ,input.shape[0],self.hidden_size)

        # 向前传播
        for i in range(self.seq_len):
            encoder_output, encoder_hidden = self.encoder(input[:,i,:], encoder_hidden)
            encoder_outputs[i] = encoder_output[0]
        """
        此处可能是对序列不敏感的原因
        """

        # encoder_outputs = encoder_output.view(input.shape[1], self.hidden_size)

        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros(input.shape[0],self.seq_len, self.output_dim)

        for i in range(self.seq_len):
            decoder_output, decoder_hidden,decoder_attention = self.decoder(
                encoder_outputs[i], decoder_hidden,encoder_outputs)
            decoder_outputs[:,i,:] =decoder_output


        return decoder_outputs[:,0,:], decoder_attention[:,0,:]



class Seq2Seq():
    def __init__(self,num,input_dim,seq_len,output_dim,hidden_size,dropout,learning_rate,batch_size,device_pu):
        self.station = num
        self.input_dim =input_dim
        self.output_dim = output_dim
        self.seq_len =seq_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lr =learning_rate
        self.bs =batch_size
        self.device =device_pu
        self.model =seq2seq_cell(self.input_dim,
                                 self.seq_len,
                                 self.output_dim,
                                 self.hidden_size,
                                 self.dropout,
                                 self.lr,
                                 self.bs,self.device).to(self.device )

    def fit(self,data,label,shuffle,num_epoches = 100):

        self.shuffle =shuffle

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

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
        uncorrect = True
        while uncorrect:
            f = open('{}/train_{}_{}_{}_{}.txt'.format(path,self.station, self.hidden_size, self.dropout, self.lr), 'w+')
            self.model =seq2seq_cell(self.input_dim,
                                self.seq_len ,
                                self.output_dim,
                                self.hidden_size,
                                self.dropout,
                                self.lr,
                                self.bs,self.device).to(self.device)
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()
            train_loss_last = np.inf
            for iter in range(1, num_epoches + 1):
                self.model.train()
                print('epoch {}'.format(iter))
                print('**************************************')
                running_loss = 0.0

                for i, data in enumerate(data_train, 1):
                    """
                    随机打乱的方式不好 应该是全部打乱之后 固定抽取 否则会出现样本利用不均衡的问题
                    """
                    img, label = data
                    img = Variable(img)
                    label = Variable(label).to(self.device)

                    img = img.view(-1,self.seq_len,self.input_dim).to(self.device)
                    decoder_output,decoder_attention =self.model(img)

                    loss = self.criterion(decoder_output, label)
                    running_loss += loss.data.item() * label.size(0)
                    # 向后传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                train_loss =running_loss / (len(y_train))
                print('Finish {} epoch, Loss: {:.6f}'.format(
                    iter, train_loss))

                if train_loss_last == train_loss and iter < 3:
                    break
                if train_loss_last > train_loss and iter >= 3:
                    uncorrect = False
                    
                train_loss_last =train_loss

                self.model.eval()
                eval_loss = 0.
                for data in data_test:
                    img, label = data

                    img = Variable(img)
                    label = Variable(label).to(self.device)

                    img = img.view( -1, self.seq_len,self.input_dim).to(self.device)

                    decoder_output,decoder_attention =self.model(img)

                    loss = self.criterion(decoder_output, label)
                    eval_loss += loss.data.item() * label.size(0)
                val_loss = eval_loss / (len(y_test))
                print('Val Loss: {:.6f}'.format(val_loss))

                
                f.write(" Train_MSE: " + str(train_loss) + ' Val_MSE: ' + str(val_loss) + '\n')

                if val_loss < eval_loss_best:
                    eval_loss_best = val_loss
                    self.eval = eval_loss_best
                    torch.save(self.model, '{}/seq2seq_{}_{}_{}_{}.pth'.format(path, self.station,self.hidden_size, self.dropout, self.lr))
            f.close()
        
        return self.eval


    def predict(self,test_data,test_label):

        test_model = torch.load('{}/seq2seq_{}_{}_{}_{}.pth'.format(path,self.station,self.hidden_size, self.dropout,self.lr)).to(self.device )

        test_loss = 0
        test = GetLoader(test_data, test_label)
        criterion = nn.MSELoss()
        data_test = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

        y_mlp = []
        y_attention=[]
        for data in data_test:

            img, label = data
            img = Variable(img).to(self.device )
            label = Variable(label).to(self.device )

            img = img.view( -1, self.seq_len,self.input_dim)

            decoder_output,decoder_attention = test_model(img)

            loss = criterion(decoder_output, label)
            test_loss += loss.data
            y_mlp.append(decoder_output.data)
            y_attention.append(decoder_attention.data.cpu().squeeze().numpy())


        print('Test Loss: {:.6f}'.format(test_loss / (len(
            test_label))))

        y_mlp = np.array(y_mlp).squeeze()[:,np.newaxis]
        y_attention =np.array(y_attention)


        return y_mlp ,y_attention





