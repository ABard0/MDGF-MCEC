import pandas as pd
import numpy as np

dis_ss = pd.read_csv('data/disease_ss.csv', header=0).values
cd_adjmat = pd.read_csv('data/assocaitions.csv', header=0).values

rows = len(cd_adjmat)
result = np.zeros([rows, rows])

for i in range(rows):
    index_list = []
    for k in range(len(cd_adjmat[1])):
       if cd_adjmat[i][k] == 1:
           index_list.append(k)
    if len(index_list) == 0:
        continue
    for j in range(0, i+1):
        index_list2 = []
        for k in range(len(cd_adjmat[1])):
            if cd_adjmat[j][k] == 1:
                index_list2.append(k)
        if len(index_list2) == 0:
            continue
        sum1=0
        sum2=0

        for k1 in range(len(index_list)):
            sum1 = sum1 + max(dis_ss(index_list[k1], index_list2))
        for k2 in range(len(index_list2)):
            sum2 = sum2 + max(dis_ss(index_list, index_list2[k2]))
        result[i, j] = (sum1 + sum2) / (len(index_list) + len(index_list2))
        result[j, i] = result[i, j]

for t in range(rows):
    result[t][t] = 1

result = pd.DataFrame(result)
result.to_csv('output/circFuncSimilarity.csv')
