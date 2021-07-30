import pandas as pd
import numpy as np
import os

record=pd.read_csv('sudeste.csv')

path ='./hour_data'
if os.path.exists(path):
    pass
else:
    os.mkdir(path)

print(record.columns)
print(record.index)

print(record.head())

print(record[record.wsid==178].head())

print(record.wsid.value_counts())

#获得数据的气象点标签
idx = record.wsid.values
idx= np.unique(idx)
idx_num=record.wsid.value_counts()

#保存数据较多的气象点 并按天得到数据的整理
for i in idx:
    if idx_num[i]>=120000:
        df=record[record.wsid==i]
    else:
        continue
    df = df.dropna(axis=0, how='all')

    df = df.fillna(value=0)

    df.to_csv('{}/hour{}.csv'.format(path,i))

print('yy')
