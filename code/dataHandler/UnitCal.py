import pandas as pd
import numpy as np

def circRNAUnitCal():
    cirR2Disease_disList = pd.read_csv('../../datasets/multiview/circRNAList.txt', header=None,
                           encoding='gb18030').values
    cir2Disease_disList = pd.read_csv('../../datasets/circ2Disease/Circ2Disease_circList.txt', header=None,
                           encoding='gb18030').values
    cirRNA2Disease_disList = pd.read_csv('../../datasets/circRNA2Disease/CircRNA2Disease_circList.txt', header=None,
                           encoding='gb18030').values
    cirR2Disease_disList = list(cirR2Disease_disList.squeeze())
    cir2Disease_disList = list(cir2Disease_disList.squeeze())
    cirRNA2Disease_disList = list(cirRNA2Disease_disList.squeeze())
    circR2disease_circ2Disease = []
    circR2disease_circRNA2Disease = []
    circ2disease_circRNA2Disease = []
    both = []

    for i in cir2Disease_disList:
        for j in cirRNA2Disease_disList:
            if (i.lower() in j.lower()) or (j.lower() in i.lower()):
                if [i, j] not in circ2disease_circRNA2Disease:
                    circ2disease_circRNA2Disease.append([i, j])

    for i in cirR2Disease_disList:
        for j in cir2Disease_disList:
            if (i.lower() in j.lower()) or (j.lower() in i.lower()):
                if [i, j] not in circR2disease_circ2Disease:
                    circR2disease_circ2Disease.append([i, j])
        for k in cirRNA2Disease_disList:
            if (i.lower() in k.lower()) or (k.lower() in i.lower()):
                if [i, k] not in circR2disease_circRNA2Disease:
                    circR2disease_circRNA2Disease.append([i, k])

    for i in cirR2Disease_disList:
        for j in cir2Disease_disList:
            for k in cirRNA2Disease_disList:
                if (j.lower() in i.lower()) and (k.lower() in i.lower()) and ((j.lower() in k.lower()) or (k.lower() in j.lower())):
                    if [i,j,k] not in both:
                        both.append([i,j,k])

    print()

def diseaseUnitCal():
    cirR2Disease_disList = pd.read_csv('../../datasets/multiview/diseaseList.txt', header=None,
                           encoding='gb18030').values
    cir2Disease_disList = pd.read_csv('../../datasets/circ2Disease/Circ2Disease_disList.txt', header=None,
                           encoding='gb18030').values
    cirRNA2Disease_disList = pd.read_csv('../../datasets/circRNA2Disease/CircRNA2Disease_disList.txt', header=None,
                           encoding='gb18030').values
    cirR2Disease_disList = list(cirR2Disease_disList.squeeze())
    cir2Disease_disList = list(cir2Disease_disList.squeeze())
    cirRNA2Disease_disList = list(cirRNA2Disease_disList.squeeze())
    circR2disease_circ2Disease = []
    circR2disease_circRNA2Disease = []
    circ2disease_circRNA2Disease = []
    both = []
    cir2Disease_disList.remove('cancer')

    for i in cir2Disease_disList:
        for j in cirRNA2Disease_disList:
            if (i.lower() in j.lower()) or (j.lower() in i.lower()):
                if [i, j] not in circ2disease_circRNA2Disease:
                    circ2disease_circRNA2Disease.append([i, j])

    for i in cirR2Disease_disList:
        for j in cir2Disease_disList:
            if (i.lower() in j.lower()) or (j.lower() in i.lower()):
                if [i, j] not in circR2disease_circ2Disease:
                    circR2disease_circ2Disease.append([i, j])
        for k in cirRNA2Disease_disList:
            if (i.lower() in k.lower()) or (k.lower() in i.lower()):
                if [i, k] not in circR2disease_circRNA2Disease:
                    circR2disease_circRNA2Disease.append([i, k])

    for i in cirR2Disease_disList:
        for j in cir2Disease_disList:
            for k in cirRNA2Disease_disList:
                if (j.lower() in i.lower()) and (k.lower() in i.lower()) and ((j.lower() in k.lower()) or (k.lower() in j.lower())):
                    if [i,j,k] not in both:
                        both.append([i,j,k])

    print()

if __name__=="__main__":
    #circRNAUnitCal()
    diseaseUnitCal()