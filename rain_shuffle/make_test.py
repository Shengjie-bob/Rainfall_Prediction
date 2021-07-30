import pandas as pd
import numpy as np
import os

path = os.getcwd()

#选择数据集
num_array=np.array([313,314,371,372,393])

#测试集的比例
ratio =0.1

for num in num_array:
    df = pd.read_csv('{}/sample/hour{}.csv'.format(path,num))

    data_num = len(df)

    test_num =int(ratio*data_num)

    test_df =df.iloc[(data_num-test_num):,1:]

    test_df.to_csv('{}/sample/station{}.csv'.format(path, num))

    print('station{}ok'.format(num))