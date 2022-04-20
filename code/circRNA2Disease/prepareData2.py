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
    c1 = []
    c2 = []
    c3 = []
    mm_matrix = read_txt('../../datasets/circRNA2Disease/multiview/CircRNA2Disease_circFSsimilarity.txt')
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
    dataset['mm_f'] = {'data_matrix': mm_matrix, 'edges': mm_edge_index}



    c1 = []
    c2 = []
    c3 = []
    mm_matrix = read_txt('../../datasets/circRNA2Disease/multiview/CircRNA2Disease_circGIPsimilarity.txt')
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
                mm_matrix[row][col] = 1.8 * mm_matrix[row][col]
    for i in range(len(c2)):
        row = c2[i]
        for j in range(len(c2)):
            col = c2[j]
            if row != col:
                mm_matrix[row][col] = 1.8 * mm_matrix[row][col]
    for i in range(len(c3)):
        row = c3[i]
        for j in range(len(c3)):
            col = c3[j]
            if mm_matrix[row][col] != 1:
                mm_matrix[row][col] = 1.8 * mm_matrix[row][col]
    mm_matrix = t.FloatTensor(mm_matrix)
    mm_edge_index = get_edge_index(mm_matrix)
    dataset['mm_g'] = {'data_matrix': mm_matrix, 'edges': mm_edge_index}


    c1 = []
    c2 = []
    c3 = []
    # dd_matrix = read_txt(opt.data_path + '/multiview/disease semantic similarity.txt')

    dd_matrix = read_txt('../../datasets/circRNA2Disease/multiview/CircRNA2Disease_disSSsimilarity.txt')
    y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(dd_matrix)
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
                dd_matrix[row][col] = 1.8 * dd_matrix[row][col]
    for i in range(len(c2)):
        row = c2[i]
        for j in range(len(c2)):
            col = c2[j]
            if row != col:
                dd_matrix[row][col] = 1.8 * dd_matrix[row][col]
    for i in range(len(c3)):
        row = c3[i]
        for j in range(len(c3)):
            col = c3[j]
            if dd_matrix[row][col] != 1:
                dd_matrix[row][col] = 1.8 * dd_matrix[row][col]
    dd_matrix = t.FloatTensor(dd_matrix)
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd_f'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}


    c1 = []
    c2 = []
    c3 = []
    dd_matrix = read_txt('../../datasets/circRNA2Disease/multiview/CircRNA2Disease_disGIPsimilarity.txt')
    y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(dd_matrix)
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
                dd_matrix[row][col] = 1.8 * dd_matrix[row][col]
    for i in range(len(c2)):
        row = c2[i]
        for j in range(len(c2)):
            col = c2[j]
            if row != col:
                dd_matrix[row][col] = 1.8 * dd_matrix[row][col]
    for i in range(len(c3)):
        row = c3[i]
        for j in range(len(c3)):
            col = c3[j]
            if dd_matrix[row][col] != 1:
                dd_matrix[row][col] = 1.8 * dd_matrix[row][col]

    dd_matrix = t.FloatTensor(dd_matrix)
    dd_edge_index = get_edge_index(dd_matrix)
    dataset['dd_g'] = {'data_matrix': dd_matrix, 'edges': dd_edge_index}
    return dataset

