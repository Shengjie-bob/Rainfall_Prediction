from sklearn.metrics import *
import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")#忽略警告
#import seaborn as sns

def evaluation(a,b):  # a为真实标签
    rmse = np.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    var = explained_variance_score(a,b)
    mdae = median_absolute_error(a,b)
    r2 =r2_score(a,b)
    return rmse, mae, mdae, r2, var


#绘图
def plot(a, b, label_a, label_b, title=None):
    plt.figure()
    if title != None:
        plt.title(str(title) )
    plt.plot(a, color = 'b', label = str(label_a))
    plt.plot(b, color = 'r', label = str(label_b))
    plt.legend(loc='best')
    plt.show()


def readandplot(filepath, save_name, title = None): # 函数用途：模型寻优过程不同超参数下的损失
    f = open(filepath,"r")
    line_all = f.readlines()
    row_num = len(line_all)
    label_list = []
    line_1 = line_all[0].strip('\n')
    a = line_1.split(' ')
    m = int(len(a) / 2)
    for i in [2*ele  for ele in range(m)]:
        label_list.append(a[i].strip(':'))
    print(label_list)
    b = np.zeros((row_num,m))
    for j,line in enumerate(line_all):
        line = line.strip('\n') 
        c = line.split(' ')
        for k in range(m):
            l = 2 * k + 1
            b[j,k] = c[l]  
    data = b
    ax = plt.figure()
    if title != None:
        plt.title(str(title) )
    colors = ['r','b']
    mark = ['v-','o-']
    if save_name == 'SVR_poly_train_371_degree':
        data = data[:-5,:]
    for i in range(m):
        plt.plot(data[:,0],data[:,i+1], mark[i],c = colors[i],alpha = 0.4,label = label_list[i+1])
        plt.xlabel(label_list[0])
        plt.legend(loc='best') 
        y_min = np.min(data[:,i+1])
        index = np.argmin(data[:,i+1])
        x_min = data[:,0][index]
        plt.text(x_min,y_min,"min",fontdict={'size':'8','color':colors[i]})
        if i==m-2:
            break
    plt.grid()
    plt.show()
    ax.savefig('output/%s.png'%save_name,bbox_inches='tight',dpi=500)



# def readTestResults(filepath):  # 函数用途：每个模型最优参数下的5个指标值对比
#     f = open(filepath,"r")
#     line_all = f.readlines()
#     row_num = len(line_all)
#     data = np.zeros((row_num,5))
#     label_list = ['ARIMA','ATT_SEQ2SEQ','GBRT','LSTM','MLP','SEQ2SEQ','SVR_poly','SVR_rbf','SVR_sigmoid','XGB']
#     for j,line in enumerate(line_all):
#         line = line.strip('\n').split(' ') 
#         for k in range(5):
#             data[j,k] = float(line[2 * k + 1])
    
#     colors = ['r','b','darkblue','cyan','violet']
    
    
#     ax = plt.figure()
#     plt.plot(data[:,0], label = 'rmse' ,c=colors[0], linestyle='--',marker='o')
#     plt.xticks(range(row_num),label_list,rotation=45) #可以是字符
#     plt.legend()
#     plt.grid()
#     plt.show()
#     ax.savefig('output/rmse.png',bbox_inches='tight',dpi=500)
    
    
#     ax = plt.figure()
#     for i,y_mark in enumerate(['mae','mdae','r2','var']):
#         i = i + 1
#         plt.plot(data[:,i], label = y_mark ,c=colors[i], linestyle='--',marker='o')
#         plt.xticks(range(row_num),label_list,rotation=45) #可以是字符
#         plt.legend()
#     plt.grid()
#     plt.show()
#     ax.savefig('output/the_other_4_metics.png',bbox_inches='tight',dpi=500)
#     return data



# import glob
# import xlwt
# import xlrd
# from xlutils.copy import copy as xl_copy
# def processTestResultsOfEachModel(filefolder,name): # 读取当前模型在所有气象站上的测试结果并保存到表格
#     dirlist = []
#     for f in glob.glob('{}/test*.txt'.format(filefolder)): # find all test*.txt files and store their paths into a list
#         dirlist.append(f)
#     station_num = len(dirlist)
#     data = np.zeros((5,station_num))
#     numid = [ele[-7:-4] for ele in dirlist]
#     for i,path in enumerate(dirlist):
#         with open(path, 'r') as f:
#             line = f.readlines()
#             line = line[0].strip('\n').split(' ')
#             data[:,i] = [float(line[2  * j + 1]) for j in range(5)]
#             if i==0:
#                 label_list = [line[2 * j][:-1] for j in range(5)]
#
#     try:
#         rb = xlrd.open_workbook('test_data_of_each_station.xls', formatting_info=True)
#         workbook = xl_copy(rb) # make a copy of it
#     except:
#
#         workbook = xlwt.Workbook(encoding='utf-8')
#     try:
#         booksheet = workbook.add_sheet('%s'%name,cell_overwrite_ok=True)
#     except:
#         print('%s already exists'%name)
#         return
#     booksheet.write(0,0,'station id')
#     for j in range(station_num):
#         booksheet.write(0,j+1,int(numid[j]))
#
#     for i in range(5):
#         booksheet.write(i+1,0,label_list[i])
#
#     for i in range(5):
#         for j in range(station_num):
#             booksheet.write(i+1,j+1,data[i,j])
#
#     booksheet.write(0,station_num+1,'mean_value')
#     for i in range(5):
#         booksheet.write(i+1,station_num+1,data[i,:].mean())
#
#     booksheet.write(0,station_num+2,'mean_value_after_eliminating_2_worst')
#     for i in range(5):
#         if i == 3:
#             booksheet.write(i+1,station_num+2,-np.sort(-data[i,:])[:-2].mean())
#         else:
#             booksheet.write(i+1,station_num+2,np.sort(data[i,:])[:-2].mean())
#
#     workbook.save('test_data_of_each_station.xls')



def plotBestResults():  # 读取表格，把每个模型的最好表现对比绘图，并保存到output文件夹
    rb = xlrd.open_workbook('test_data_of_each_station.xls', formatting_info=True)
    sheetNames = rb.sheet_names() #获取所有sheet的名字，sheetNames为list类型
    a,b,c,d,e = [], [], [], [], []
    
    model_name_list = []
    for sheet in sheetNames:
        table = rb.sheet_by_name(sheet)
        a.append(table.row_values(1)[-1])
        b.append(table.row_values(2)[-1])
        c.append(table.row_values(3)[-1])
        d.append(table.row_values(4)[-1])
        e.append(table.row_values(5)[-1])
        model_name_list.append(table.row_values(1)[0].split('_')[0])
    xmark = model_name_list
    num = len(xmark)
    plt.figure();plt.plot(a,'o-',c='salmon', label = 'rmse');plt.legend();plt.xticks(range(num),xmark,rotation=45);plt.grid();
    plt.savefig('output/rmse.png',bbox_inches='tight',dpi = 500);plt.show()
    plt.figure();plt.plot(b,'*-',c='limegreen', label = 'mae');plt.legend();plt.xticks(range(num),xmark,rotation=45);
    plt.plot(c, 'v-',c='blue',label = 'mdae');plt.legend();plt.xticks(range(num),xmark,rotation=45);
    plt.plot(d, '^-',c='r',label = 'r2_score');plt.legend();plt.xticks(range(num),xmark,rotation=45);
    plt.plot(e,'D-',c='darkorchid', label = 'var');plt.legend();plt.xticks(range(num),xmark,rotation=45);plt.grid();
    plt.savefig('output/the_other_4_merics.png',bbox_inches='tight',dpi = 500)
    plt.show()
    


import joblib
import torch
from torch import Tensor
from torch.autograd import Variable
def plotPredictionAndTruth(path, model_name, model_folder, num,test_x, test_y):  # 模型预测与真实可视化
    if model_name.split('.')[-1] =='m':
        clf = joblib.load(path)
        y_pre_temp = clf.predict(test_x)
    else:
        if model_name[:3]=='mlp':
            clf = torch.load(path,map_location=torch.device('cpu'))
            x_tensor = Tensor(test_x)
            x_tensor = Variable(x_tensor)
            y_pre_temp = clf(x_tensor).detach().numpy()          
        else:
            clf = torch.load(path,map_location=torch.device('cpu'))
            x_tensor = Tensor(test_x)
            x_tensor = Variable(x_tensor).view(-1,7,3)
            y_pre_temp = clf(x_tensor).detach().numpy()
    
    rmse, mae, mdae, r2, var = evaluation(test_y, y_pre_temp)   
    y_pre_temp = y_pre_temp.reshape(1,-1)  
    test_y = test_y.reshape(1,-1)
    # 绘图
    plt.figure()
    plt.plot(test_y.squeeze() ,c='r', label = 'true value')
    plt.plot(y_pre_temp.squeeze() ,c='b', alpha = 0.5, label = 'prediction value')
    y_max = max(np.append(test_y,y_pre_temp))
    dy = 2
    plt.text(0, y_max, "model:%s"%model_folder, size = 10,color = "black", style = "italic", weight = "light")
    plt.text(0, y_max- dy, "station:%d"%num, size = 10,color = "black", style = "italic", weight = "light")
    plt.text(0, y_max-2*dy, "rmse:%.3f"%rmse, size = 10,color = "black", style = "italic", weight = "light")
    plt.text(0, y_max-3 * dy, "mae:%.3f"%mae, size = 10,color = "black", style = "italic", weight = "light")
    plt.text(0, y_max-4 * dy, "mdae:%.3f"%mdae, size = 10,color = "black", style = "italic", weight = "light")
    plt.text(0, y_max-5 * dy, "r2:%.3f"%r2, size = 10,color = "black", style = "italic", weight = "light")
    plt.text(0, y_max-6 * dy, "var:%.3f"%var, size = 10,color = "black", style = "italic", weight = "light")

    plt.legend()
    plt.savefig('output/visualization_%s.png'%model_name.split('.')[0],bbox_inches='tight',dpi = 500)
    plt.show()


def netModelFindBest(filepath):
    name = filepath.split('/')[-1]   
    txt_list = glob.glob('{}/train_371*.txt'.format(filepath))
    num = len(txt_list) 
    min_val_loss_list = []
    hyp_list = []
    for i in range(num):
        # hyp_list = txt_list[i].split('\\')[-1].split('_')[2:]
        # hyp_list[-1] = hyp_list[-1].split('.txt')[0]
        # hyp_list = [float(ele) for ele in hyp_list]
        hyp = txt_list[i].split('\\')[-1].strip('.txt')[9:]
        f = open(txt_list[i],'r')
        line_all = f.readlines()
        min_val_loss = 100
        for line in line_all:
            
            temp_val_loss = float(line_all[0].strip('\n').split()[-1])
            if temp_val_loss < min_val_loss:
                min_val_loss = temp_val_loss
        min_val_loss_list.append(min_val_loss)
        hyp_list.append(hyp)
    
    plt.figure()
    plt.plot(min_val_loss_list, 'v-',c='blue',alpha = 0.5,label = 'val_loss')
    plt.legend()
    plt.xticks(range(num),hyp_list,rotation=45)
    plt.grid()
    y_min = np.min(min_val_loss_list)
    x_min = np.argmin(min_val_loss_list)
    plt.text(x_min,y_min,"min",fontdict={'size':'8','color':'b'})
    plt.savefig('output/find_best_hyp_%s.png'%name,bbox_inches='tight',dpi = 500)
    plt.show()
    




if __name__ == '__main__':
    pathnow = os.getcwd()
    
    # .m模型模型的寻优过程，图保存在output文件夹里
    # filepath = '{}/SVR_poly/train_371_C.txt'.format(pathnow)
    # filepath2 = '{}/SVR_poly/train_371_degree.txt'.format(pathnow)
    # filepath3 = '{}/SVR_poly/train_371_gamma.txt'.format(pathnow)
    # filepath4 = '{}/GBRT/train_371_depth.txt'.format(pathnow)
    # filepath5 = '{}/GBRT/train_371_est.txt'.format(pathnow)
    # filepath6 = '{}/GBRT/train_371_lr.txt'.format(pathnow)
    # filepath7 = '{}/XGB/train_371_depth.txt'.format(pathnow)
    # filepath8 = '{}/XGB/train_371_est.txt'.format(pathnow)
    # filepath9 = '{}/XGB/train_371_lr.txt'.format(pathnow)
    # filepath10 = '{}/SVR_sigmoid/train_371_C.txt'.format(pathnow)
    # filepath11 = '{}/SVR_sigmoid/train_371_gamma.txt'.format(pathnow)
    # filepath12 = '{}/SVR_rbf/train_371_C.txt'.format(pathnow)
    # filepath13 = '{}/SVR_rbf/train_371_gamma.txt'.format(pathnow)
    # readandplot(filepath,save_name = 'SVR_poly_train_371_C',title = None)
    # readandplot(filepath2,save_name = 'SVR_poly_train_371_degree')  
    # readandplot(filepath3,save_name = 'SVR_poly_train_371_gamma') 
    # readandplot(filepath4,save_name = 'GBRT_train_371_depth') 
    # readandplot(filepath5,save_name = 'GBRT_train_371_est') 
    # readandplot(filepath6,save_name = 'GBRT_train_371_lr') 
    # readandplot(filepath7,save_name = 'XGB_train_371_depth') 
    # readandplot(filepath8,save_name = 'XGB_train_371_est') 
    # readandplot(filepath9,save_name = 'XGB_train_371_lr') 
    # readandplot(filepath10,save_name = 'SVR_sigmoid_train_371_C') 
    # readandplot(filepath11,save_name = 'SVR_sigmoid_train_371_gamma') 
    # readandplot(filepath12,save_name = 'SVR_rbf_train_371_C') 
    # readandplot(filepath13,save_name = 'SVR_rbf_train_371_gamma') 
    
    
    
    
    # 读取每个模型在所有气象站上的测试结果并保存到表格，
    # 再读取表格，把每个模型的最好表现对比绘图，并保存到output文件夹
    # name_list = ['ARIMA','ATT_SEQ2SEQ','GBRT','LSTM','MLP','SEQ2SEQ','SVR_rbf','XGB']
    # for name in name_list:
    #     filefolder = '{}/{}'.format(pathnow,name)
    #     processTestResultsOfEachModel(filefolder,name)
    # plotBestResults()
    
    
    
       
    # 目标值与预测值可视化
    # model_folder = 'MLP'
    
    # path_2 = 'train_test_hour'
    
    # from sklearn.preprocessing import *
    # path = pathnow   
    # num_list = [312,313,314,315,316,371,372,373,374,393,394,396]
    # for num in num_list:
    #     try:
    #         path_to_model = glob.glob('{}/{}/*{}*.m'.format(pathnow,model_folder,num)) [0]
    #     except:
    #         path_to_model = glob.glob('{}/{}/*{}*.pth'.format(pathnow,model_folder,num))[0]
    #     model_name = path_to_model.split('\\')[-1]
    #     test_feature = np.loadtxt("{}/{}/test{}.txt".format(path,path_2, num)).astype(np.float32)
    #     test_rain = np.loadtxt("{}/{}/test_label{}.txt".format(path,path_2, num)).astype(np.float32)
    #     test_feature = scale(test_feature, axis=0)
    #     test_rain = np.reshape(test_rain, (-1, 1))
    #     plotPredictionAndTruth(path_to_model,  model_name, model_folder, num, test_feature, test_rain)




    # .pth文件的寻优过程
    # filepath = '{}/MLP'.format(pathnow)
    # netModelFindBest(filepath)
    
    
    
    
    # ATT_SEQ2SEQ的可视化
    # num = 312
    # path_2 = 'train_test_hour'
    # test_feature = np.loadtxt("{}/{}/test{}.txt".format(pathnow,path_2, num)).astype(np.float32)
    # test_rain = np.loadtxt("{}/{}/test_label{}.txt".format(pathnow,path_2, num)).astype(np.float32)
    # from sklearn.preprocessing import *
    # test_feature = scale(test_feature, axis=0)
    # test_rain = np.reshape(test_rain, (-1, 1))
    
    # model = torch.load('{}\ATT_SEQ2SEQ\seq2seq_312_128_0.1_0.01.pth'.format(pathnow),map_location=torch.device('cpu'))
    # x_tensor = Tensor(test_feature)
    # # x_tensor = Variable(x_tensor).view(-1,7,3)
    # x_tensor = Variable(x_tensor).view(-1,3,7)
    # y_seq2seq,y_attention = model(x_tensor)
    
    

    
    
    # plt.figure()
    # plt.plot(test_rain.squeeze() ,c='r', label = 'true value')
    # plt.plot(y_seq2seq.squeeze() ,c='b', alpha = 0.5, label = 'prediction value')
    # y_max = max(np.append(test_rain,y_seq2seq.detach().numpy()))
    # dy = 2
    # plt.text(0, y_max, "model:%s"%model_folder, size = 10,color = "black", style = "italic", weight = "light")
    # plt.text(0, y_max- dy, "station:%d"%num, size = 10,color = "black", style = "italic", weight = "light")
    # plt.text(0, y_max-2*dy, "rmse:%.3f"%rmse, size = 10,color = "black", style = "italic", weight = "light")
    # plt.text(0, y_max-3 * dy, "mae:%.3f"%mae, size = 10,color = "black", style = "italic", weight = "light")
    # plt.text(0, y_max-4 * dy, "mdae:%.3f"%mdae, size = 10,color = "black", style = "italic", weight = "light")
    # plt.text(0, y_max-5 * dy, "r2:%.3f"%r2, size = 10,color = "black", style = "italic", weight = "light")
    # plt.text(0, y_max-6 * dy, "var:%.3f"%var, size = 10,color = "black", style = "italic", weight = "light")

    # plt.legend()
    # plt.savefig('output/visualization_%s.png'%model_name.split('.')[0],bbox_inches='tight',dpi = 500)
    # plt.show()



    # fig =plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(y_attention[:5,:], cmap='bone')
    # fig.colorbar(cax)
    # plt.show()
    
    # path ='./ATT_SEQ2SEQ'
    # f = open('{}/test_ATT_SEQ2SEQ_{}.txt'.format(path,num), 'w+')
    # f.write('ATT_SEQ2SEQ_rmse: %r ' % rmse +
    #       'ATT_SEQ2SEQ_mae: %r ' % mae +
    #       'ATT_SEQ2SEQ_mdae: %r ' % mdae +
    #       'ATT_SEQ2SEQ_r2: %r ' % r2 +
    #       'ATT_SEQ2SEQ_var: %r ' % var)
    # f.close()


