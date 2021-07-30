import pandas as pd
import numpy as np
import os

record=pd.read_csv('sudeste.csv')

path ='./day_data'
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
    if idx_num[i]>=100000:
        df=record[record.wsid==i]
    else:
        continue
    df = df.dropna(axis=0, how='all')

    df = df.fillna(value=0)

    tim = df.date.values

    tim = np.unique(tim)

    k = 0
    # 得到每天的降水数据
    for j in tim:
        k = k + 1
        df1 = df[df.date == j]
        djsk = df1.values
        data = np.mean(djsk[:, 14:30], axis=0)
        rain = np.sum(djsk[:, 14])
        djsk = djsk[-1, :][:, np.newaxis].T
        djsk[:, 14:30] = data
        djsk[:, 14] = rain

        if k == 1:
            df_all = djsk
        else:
            df_all = np.vstack([df_all, djsk])

    df = pd.DataFrame(df_all)

    df.to_csv('{}/tian{}.csv'.format(path,i))


print('yy')
