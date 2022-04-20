import numpy
import numpy as np
import pandas as pd

def Circ2DiseaseHanddle():
    io = '../../datasets/Circ2Disease.xlsx'
    data = pd.read_excel(io, engine='openpyxl')
    circ = list(set(data['circ']))
    dis = list(set(data['dis']))
    association = numpy.zeros((len(circ), len(dis)))
    for i, row in data.iterrows():
        c_index = circ.index(row["circ"])
        d_index = dis.index(row["dis"])
        association[c_index, d_index] = 1
    pd_data = pd.DataFrame(association)
    pd_data.to_csv('../../datasets/circ2Disease/Circ2Disease_Association.csv', header=False, index=False)

    pd_cirli = pd.DataFrame(circ)
    pd_cirli.to_csv('../../datasets/circ2Disease/Circ2Disease_circList.txt', header=False, index=False)
    pd_disli = pd.DataFrame(dis)
    pd_disli.to_csv('../../datasets/circ2Disease/Circ2Disease_disList.txt', header=False, index=False)
    print()

def CircRNA2DiseaseHanddle():
    io = '../../datasets/CircRNA2Disease.xlsx'
    data = pd.read_excel(io, engine='openpyxl')
    circ = []
    dis = []
    for i, row in data.iterrows():
        if row["circ_id"] == '-':
            if row["circ_name"] != '-':
                circ.append(row["circ_name"])
            else:
                print(i)
        else:
            circ.append(row["circ_id"])
        dis.append(row["dis"])
    circ = list(set(circ))
    dis = list(set(dis))
    association = numpy.zeros((len(circ), len(dis)))
    for i, row in data.iterrows():
        if row["circ_id"] == '-':
            if row["circ_name"] != '-':
                c_index = circ.index(row["circ_name"])
            else:
                print(i)
        else:
            c_index = circ.index(row["circ_id"])
        d_index = dis.index(row["dis"])
        association[c_index, d_index] = 1
    pd_data = pd.DataFrame(association)
    pd_data.to_csv('../../datasets/circRNA2Disease/CircRNA2Disease_Association.csv', header=False, index=False)

    pd_cirli = pd.DataFrame(circ)
    pd_cirli.to_csv('../../datasets/circRNA2Disease/CircRNA2Disease_circList.txt', header=False, index=False)
    pd_disli = pd.DataFrame(dis)
    pd_disli.to_csv('../../datasets/circRNA2Disease/CircRNA2Disease_disList.txt', header=False, index=False)
    print()


if __name__=="__main__":
    # handdle Circ2Disease data process
    # Circ2DiseaseHanddle()
    # handdle CircRNA2Disease data process
    CircRNA2DiseaseHanddle()