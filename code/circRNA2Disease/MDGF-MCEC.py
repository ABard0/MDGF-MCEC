

# -*- coding: utf-8 -*-
###THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python
import numpy as np
import pandas as pd
import os

import torch
from matplotlib import pyplot
from numpy import interp

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import model_selection

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io

from keras.layers import merge


from keras.utils import np_utils, generic_utils

from xgboost import XGBClassifier
#from keras.layers import containers, normalization

from prepareData2 import prepare_data
from model import GCNModel
from trainData import Dataset
from torch import nn, optim
from param import parameter_parser

'''
label1是positive样本label
label2是未知样本的label
'''
def prepare_data2(mv, seperate=False):
    print("loading preResult")

    circPath = "../../datasets/circRNA2Disease/Embedding/ciRNAEmbed"+str(mv)+".csv"
    disPath = "../../datasets/circRNA2Disease/Embedding/diseaseEmbed"+str(mv)+".csv"
    circRNA_fea = np.loadtxt(circPath, dtype=float, delimiter=",")
    disease_fea = np.loadtxt(disPath, dtype=float, delimiter=",")


    # circRNA_fea = np.loadtxt('../datasets/embed2/ciRNAEmbed.csv',dtype=float,delimiter=",")
    # disease_fea = np.loadtxt('../datasets/embed2/diseaseEmbed.csv',dtype=float,delimiter=",")
    # disease_fea2 = np.loadtxt("../datasets/integrated Disease Similarity.csv",dtype=float,delimiter=",")
    # circRNA_fea2  = np.loadtxt("../datasets/integrated CircRNA  Similarity.csv",dtype=float,delimiter=",")
    interaction = np.loadtxt("../../datasets/circRNA2Disease/CircRNA2Disease_Association.csv",dtype=float,delimiter=",")




    link_number = 0
    #nonlink_number=0
    train = []
    testfnl= []
    label1 = []
    label2 = []
    label22=[]
    ttfnl=[]
    #link_position = []
    #nonLinksPosition = []

    for i in range(0, interaction.shape[0]):   # shape[0] returns m if interaction is m*n, ie, returns no. of rows of matrix
        for j in range(0, interaction.shape[1]):

            if interaction[i, j] == 1:                      #for associated
                label1.append(interaction[i,j])             #label1= labels for association(1)
                link_number = link_number + 1               #no. of associated samples
                #link_position.append([i, j])
                circRNA_fea_tmp = list(circRNA_fea[i])
                disease_fea_tmp = list(disease_fea[j])
                tmp_fea = (circRNA_fea_tmp,disease_fea_tmp)   #concatnated feature vector for an association
                train.append(tmp_fea)                       #train contains feature vectors of all associated samples
            elif interaction[i,j] == 0:                     #for no association
                label2.append(interaction[i,j])             #label2= labels for no association(0)
                #nonlink_number = nonlink_number + 1
                #nonLinksPosition.append([i, j])
                circRNA_fea_tmp1 = list(circRNA_fea[i])
                disease_fea_tmp1 = list(disease_fea[j])
                test_fea= (circRNA_fea_tmp1,disease_fea_tmp1) #concatenated feature vector for not having association
                testfnl.append(test_fea)                    #testfnl contains feature vectors of all non associated samples


    print("link_number",link_number)

    m = np.arange(len(label2))
    np.random.shuffle(m)

    for x in m:
        ttfnl.append(testfnl[x])
        label22.append(label2[x])

    for x in range(0, link_number):                         #for equalizing positive and negative samples
        tfnl= ttfnl[x]                                    #tfnl= feature vector pair for no association
        lab= label22[x]                                      #lab= label of the above mentioned feature vector pair(0)
        #print(tfnl)
        #print('***')
        train.append(tfnl)                                  #append the non associated feature vector pairs to train till x<=no. of associated pairs
        label1.append(lab)                                   #append the labels of non associated pairs(0) to label1

    return np.array(train), label1, np.array(testfnl)

def prepare_preData(seperate = False):
    print("loading preResult")

    circRNA_fea = np.loadtxt("../datasets/embed2/ciRNAEmbed.csv", dtype=float, delimiter=",")
    disease_fea = np.loadtxt("../datasets/embed2/diseaseEmbed.csv", dtype=float, delimiter=",")
    interaction = np.loadtxt("../datasets/Circ2Disease_Association.csv", dtype=float, delimiter=",")

    link_number = 0
    # nonlink_number=0
    train = []
    testfnl = []
    label1 = []
    label2 = []
    label22 = []
    ttfnl = []
    # link_position = []
    nonLinksPosition = []

    for i in range(0, interaction.shape[0]):  # shape[0] returns m if interaction is m*n, ie, returns no. of rows of matrix
        for j in range(0, interaction.shape[1]):
            if interaction[i, j] == 0:  # for no association
                label2.append(interaction[i, j])  # label2= labels for no association(0)
                # nonlink_number = nonlink_number + 1
                nonLinksPosition.append([i, j])
                circRNA_fea_tmp1 = list(circRNA_fea[i])
                disease_fea_tmp1 = list(disease_fea[j])
                test_fea = (circRNA_fea_tmp1, disease_fea_tmp1)  # concatenated feature vector for not having association
                testfnl.append(test_fea)  # testfnl contains feature vectors of all non associated samples


    m = np.arange(len(label2))

    for x in m:
        ttfnl.append(testfnl[x])
    # print('************')
    # print(ttfnl)
    # print('************')
    # print(label22)

    # print(train)
    # print(label1)
    return np.array(ttfnl), nonLinksPosition, np.array(testfnl)

def calculate_performace(test_num, pred_y,  labels): #pred_y = proba, labels = real_labels
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1

    acc = float(tp + tn)/test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score=0
        sensitivity =  float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
    else:
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        f1_score= float(2*tp)/((2*tp)+fp+fn)

    return acc, precision, sensitivity, specificity, MCC,f1_score

def transfer_array_format(data):    #preResult=X  , X= all the miRNA features, disease features
    formated_matrix1 = []
    formated_matrix2 = []
    #pdb.set_trace()
    #pdb.set_trace()
    for val in data:
        #formated_matrix1.append(np.array([val[0]]))
        formated_matrix1.append(val[0])   #contains miRNA features ?
        formated_matrix2.append(val[1])   #contains disease features ?
        #formated_matrix1[0] = np.array([val[0]])
        #formated_matrix2.append(np.array([val[1]]))
        #formated_matrix2[0] = val[1]

    return np.array(formated_matrix1), np.array(formated_matrix2)

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder




class Config(object):
    def __init__(self):
        self.data_path = '../../datasets'
        self.validation = 1
        self.save_path = '../preResult'
        #self.epoch = 300
        self.epoch = 100
        self.alpha = 0.2

class Sizes(object):
    def __init__(self, dataset):
        self.m = dataset['mm']['preResult'].size(0)
        self.d = dataset['dd']['preResult'].size(0)
        self.fg = 256
        self.fd = 256
        self.k = 32



def train(model, train_data, optimizer, opt):
    model.train()
    # regression_crit = Myloss()
    # one_index = train_data[2][0FS].cuda().t().tolist()
    # zero_index = train_data[2][1].cuda().t().tolist()

    def train_epoch():
        model.zero_grad()
        score, ciRNAEmbed, disEmbed = model(train_data)
        loss = torch.nn.MSELoss(reduction='mean')
        loss = loss(score, train_data['md_p'].cuda())

        loss.backward()
        optimizer.step()
        return loss
    def getEmbedding():
        model.zero_grad()
        score, ciRNAEmbed, disEmbed = model(train_data)
        return ciRNAEmbed, disEmbed

    for epoch in range(1, opt.epoch+1):
        train_reg_loss = train_epoch()
        print(train_reg_loss.item())
    ciRNAEmbed, disEmbed = getEmbedding()
    return ciRNAEmbed, disEmbed



opt = Config()

def prediction():
    X2, index, T = prepare_preData(seperate = True)
    X_data1, X_data2 = transfer_array_format(X2)  # X-data1 = miRNA features(2500*495),  X_data2 = disease features (2500*383)

    print("************")
    print(X_data1.shape, X_data2.shape)  # (36352,512), (36352,71)
    print("******************")

    X_data1 = np.concatenate((X_data1, X_data2), axis=1)  # axis=1 , rowwoise concatenation

    print("************")
    print(X_data1.shape)  # (36352,583)
    print("******************")

    #encoder, X_data1 = DNN_auto2(X_data1)  # Now X_data1 contains Auto encoded output

    return X_data1, index




def MDGF_MCEC():
    # 以下为GCN的Embedding过程
    # args = parameter_parser()
    # dataset = prepare_data(opt)
    # train_data = dataset
    #
    # for k in range(1, 10):
    #     for i in range(opt.validation):
    #         print('-' * 50)
    #         model = GCNModel(args, k)
    #         model.cuda()
    #         optimizer = optim.Adam(model.parameters(), lr=0.001)
    #         ciRNAEmbed, disEmbed = train(model, train_data, optimizer, args)
    #         print()
    #     ciRNAEmbed = ciRNAEmbed.detach().cpu().numpy()
    #     diseaseEmbed = disEmbed.detach().cpu().numpy()
    #     circPath = '../../datasets/circRNA2Disease/Embedding/ciRNAEmbed'+str(k)+'.csv'
    #     disPath = '../../datasets/circRNA2Disease/Embedding/diseaseEmbed'+str(k)+'.csv'
    #     np.savetxt(circPath, ciRNAEmbed, delimiter=',')
    #     np.savetxt(disPath, diseaseEmbed, delimiter=',')
    # print()



    # for i in range(opt.validation):
    #     print('-' * 50)
    #     model = GCNModel(args)
    #     model.cuda()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)
    #     ciRNAEmbed, disEmbed= train(model, train_data, optimizer, args)
    #     print()
    # ciRNAEmbed = ciRNAEmbed.detach().cpu().numpy()
    # diseaseEmbed = disEmbed.detach().cpu().numpy()
    #
    # np.savetxt('../datasets/embed2/ciRNAEmbed.csv', ciRNAEmbed, delimiter=',')
    # np.savetxt('../datasets/embed2/diseaseEmbed.csv', diseaseEmbed, delimiter=',')
    #
    # print()


    X = []
    labels = []

    for j in range(1, 10):
        X1, labels1, T = prepare_data2(j, seperate=True)
        X.append(X1)
        labels.append(labels1)



    #X= array of concatinated features,labels=corresponding labels

    #X2, index, T = prepare_preData(seperate = True)
    #import pdb            #python debugger
    X_data1 = []
    X_data2 = []
    y = []
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    num = np.arange(540)  # num gets an array like num = [0,1,2...len(y)], len(y) = 512*71 = 36352
    np.random.shuffle(num)
    for j in range(9):

        X_data1_, X_data2_ = transfer_array_format(X[j]) # X-data1 = miRNA features(2500*495),  X_data2 = disease features (2500*383)
        X_data1_ = np.concatenate((X_data1_, X_data2_ ), axis = 1) #axis=1 , rowwoise concatenation
        y_, encoder = preprocess_labels(labels[j])#  labels labels_new
        X_data1_ = X_data1_[num]
        X_data2_ = X_data2_[num]
        y_ = y_[num]
        t=0
        X_data1.append(X_data1_)
        X_data2.append(X_data2_)
        y.append(y_)

    num_cross_val = 5

    all_performance = []

    all_prob = {}
    num_classifier = 3
    all_prob[0] = []
    all_prob[1] = []
    all_prob[2] = []
    all_prob[3] = []
    all_averrage = []

    for fold in range(num_cross_val):
        #五折划分数据
        for i in range(9):
            #每折的数据分别训练9个分类器
            a=0

    for fold in range(num_cross_val):
        # 每一折的9种分类器数据
        train1 = []
        test1 = []
        train_label = []
        test_label = []
        realLabel = []
        trainLabelNew = []
        probaList = []
        probaCoefList = []

        for i in range(9):
            trainTmp = np.array([x for i, x in enumerate(X_data1[i]) if i % num_cross_val != fold])
            testTmp = np.array([x for i, x in enumerate(X_data1[i]) if i % num_cross_val == fold])
            #train2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val != fold])
            #test2 = np.array([x for i, x in enumerate(X_data2) if i % num_cross_val == fold])
            train_labelTmp = np.array([x for i, x in enumerate(y[i]) if i % num_cross_val != fold])
            test_labelTmp = np.array([x for i, x in enumerate(y[i]) if i % num_cross_val == fold])

            # trainTmp2 = X_data1[i][:1040]
            # testTmp2 = X_data1[i][1040:]
            # train_labelTmp2 = y[i][:1040]
            # test_labelTmp2 = y[i][1040:]

            train1.append(trainTmp)
            test1.append(testTmp)
            train_label.append(train_labelTmp)
            test_label.append(test_labelTmp)

        clfName=''
        #分类
        for i in range(9):
            real_labelTmp = []
            for val in test_label[i]:
                if val[0] == 1:  # tuples in array, val[0]- first element of tuple
                    real_labelTmp.append(0)
                else:
                    real_labelTmp.append(1)
            train_label_newTmp = []
            for val in train_label[i]:
                if val[0] == 1:
                    train_label_newTmp.append(0)
                else:
                    train_label_newTmp.append(1)
            class_index = 0
            prefilter_train = train1[i]
            prefilter_test = test1[i]
            clf = XGBClassifier(n_estimators=10, max_depth=3)

            #clf = RandomForestClassifier(n_estimators=100, max_depth=6)
            clf.fit(prefilter_train, train_label_newTmp)  # ***Training
            ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # **testing

            proba = transfer_label_from_prob(ae_y_pred_prob)
            probaList.append(proba)
            probaCoefList.append(ae_y_pred_prob)
            realLabel.append(real_labelTmp)
            trainLabelNew.append(train_label_newTmp)

        # 单独一折求平均
        avgProbCoef = probaCoefList[0]
        for i in range(1, 9):
            tempProb = probaCoefList[i]
            for j in range(len(avgProbCoef)):
                avgProbCoef[j] = avgProbCoef[j]+tempProb[j]
        for i in range(len(avgProbCoef)):
            avgProbCoef[i] = avgProbCoef[i]/9

        avgProb = transfer_label_from_prob(avgProbCoef)
        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(realLabel[1]), avgProb,
                                                                                       realLabel[1])
        # avgProb = transfer_label_from_prob(probaCoefList[9])
        # acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(realLabel[1]), avgProb,
        #                                                                                realLabel[1])

        fpr, tpr, auc_thresholds = roc_curve(realLabel[1], avgProbCoef)
        auc_score = auc(fpr, tpr)
        scipy.io.savemat('raw_DNN',{'fpr':fpr,'tpr':tpr,'auc_score':auc_score})

        precision1, recall, pr_threshods = precision_recall_curve(realLabel[1], avgProbCoef)

        # pyplot.plot(recall, precision1, label= 'ROC fold %d (AUC = %0.4f)' % (t, auc_score))

        aupr_score = auc(recall, precision1)
        print ("AUTO-RF:", acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score,f1_score)
        all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score,f1_score])
        t =t+1  #  AUC fold number

        #pyplot.plot(fpr, tpr, label='%s(AUC = %0.4f)' % (clfName, auc_score))
        pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))
        mean_tpr += interp(mean_fpr, fpr, tpr) # one dimensional interpolation
        mean_tpr[0] = 0.0

        pyplot.xlabel('False positive rate, (1-Specificity)')
        pyplot.ylabel('True positive rate,(Sensitivity)')
        pyplot.title('Receiver Operating Characteristic curve: 5-Fold CV')
        #pyplot.title('Five classification method comparision')

        #以下为预测
        # if fold == num_cross_val-1:
        #     high_prob = []
        #     high_index = []
        #     pre_data, index = prediction()
        #     pre_pred_prob = clf.predict_proba(pre_data)[:, 1]  # **testing
        #     for i in range(len(index)):
        #         if pre_pred_prob[i] > 0.5:
        #             high_index.append(index[i])
        #             high_prob.append(pre_pred_prob[i])
        #     print()
        #     diseaseList = []
        #     with open('../preResult/Circ2Disease_disList.txt', 'r') as f:
        #         for line in f.readlines():
        #             line = line.strip()
        #             diseaseList.append(line)
        #     circList = []
        #     with open('../preResult/Circ2Disease_circList.txt', 'r') as f:
        #         for line in f.readlines():
        #             line = line.strip()
        #             circList.append(line)
        #     pre_result = []
        #     for i in range(len(high_index)):
        #         ind = high_index[i]
        #         cir = ind[0]
        #         dis = ind[1]
        #         circRNA = circList[cir]
        #         disease = diseaseList[dis]
        #         pre_result.append([circRNA, disease, high_prob[i]])
        #     df = pd.DataFrame(pre_result)
        #     df.to_csv("../preResult/predictResult2.csv")
        #     print()

    mean_tpr /= num_cross_val
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    #
    #
    #
    #
    pyplot.plot(mean_fpr, mean_tpr,'--' ,linewidth=2.5,label='Mean ROC (AUC = %0.4f)' % mean_auc)
    pyplot.legend()


    pyplot.show()




    print('*******AUTO-RF*****')
    print ('mean performance of rf using raw feature')
    print (np.mean(np.array(all_performance), axis=0))
    Mean_Result=[]
    Mean_Result= np.mean(np.array(all_performance), axis=0)
    print ('---' * 20)
    print('Mean-Accuracy=', Mean_Result[0],'\n Mean-precision=',Mean_Result[1])
    print('Mean-Sensitivity=', Mean_Result[2], '\n Mean-Specificity=',Mean_Result[3])
    print('Mean-MCC=', Mean_Result[4],'\n' 'Mean-auc_score=',Mean_Result[5])
    print('Mean-Aupr-score=', Mean_Result[6],'\n' 'Mean_F1=',Mean_Result[7])
    print ('---' * 20)

    #print(X_data1.shape)




def transfer_label_from_prob(proba):
    label = [1 if val>=0.5 else 0 for val in proba]
    return label






if __name__=="__main__":
    MDGF_MCEC()

