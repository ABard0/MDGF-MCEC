import csv
import torch as t
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        # return t.FloatTensor(md_data)
        return md_data


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)




def prepare_data(opt):
    dataset = dict()
    dataset['md_p'] = read_csv('../../datasets/circRNA2Disease/CircRNA2Disease_Association.csv')
    dataset['md_true'] = read_csv('../../datasets/circRNA2Disease/CircRNA2Disease_Association.csv')

    zero_index = []
    one_index = []
    for i in range(dataset['md_p'].size(0)):
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = t.LongTensor(zero_index)
    one_tensor = t.LongTensor(one_index)
    dataset['md'] = dict()
    dataset['md']['train'] = [one_tensor, zero_tensor]

    def createSimilarityInfo(Dataset=dataset, path='', name=''):
        c1 = []
        c2 = []
        c3 = []
        mm_matrix = read_txt(path)
        y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(mm_matrix)
        y_pred = list(y_pred)

        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                c1.append(i)
            elif y_pred[i] == 1:
                c2.append(i)
            else:
                c3.append(i)
        for i in range(len(c1)):
            row = c1[i]
            for j in range(len(c1)):
                col = c1[j]
                if row != col:
                    mm_matrix[row][col] = 1.8*mm_matrix[row][col]
        for i in range(len(c2)):
            row = c2[i]
            for j in range(len(c2)):
                col = c2[j]
                if row != col:
                    mm_matrix[row][col] = 1.8*mm_matrix[row][col]
        for i in range(len(c3)):
            row = c3[i]
            for j in range(len(c3)):
                col = c3[j]
                if mm_matrix[row][col] != 1:
                    mm_matrix[row][col] = 1.8*mm_matrix[row][col]
        mm_matrix = t.FloatTensor(mm_matrix)
        mm_edge_index = get_edge_index(mm_matrix)
        Dataset[name] = {'data_matrix': mm_matrix, 'edges': mm_edge_index}

    createSimilarityInfo(dataset, '../../datasets/circ2Disease/multiview/CircRNA2Dasease_circFSsimilarity.txt', 'mm_f')
    createSimilarityInfo(dataset, '../../datasets/circ2Disease/multiview/CircRNA2Dasease_circGIPsimilarity.txt', 'mm_g')
    createSimilarityInfo(dataset, '../../datasets/circ2Disease/multiview/CircRNA2Dasease_disSSsimilarity.txt', 'dd_f')
    createSimilarityInfo(dataset, '../../datasets/circ2Disease/multiview/CircRNA2Dasease_disGIPsimilarity.txt', 'dd_g')

    return dataset


