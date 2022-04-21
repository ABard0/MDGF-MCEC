import math
import numpy as np
import pandas as pd


meshid = pd.read_csv('data/MeSHID.csv', header=0)
disease = meshid['disease'].tolist()
id = meshid['ID'].tolist()

meshdis = pd.read_csv('data/Mesh_disease.csv', header=0)
unique_disease = meshdis['C1'].tolist()


disease_list = []
fullID = []

for i in range(len(id)):

    disease_family = [disease[i], id[i]]
    fullID.append(id[i])
    if len(id[i]) > 3:
        id[i] = id[i][:-4]
        disease_family.append(id[i])
        fullID.append(id[i])
        if len(id[i]) > 3:
            id[i] = id[i][:-4]
            disease_family.append(id[i])
            fullID.append(id[i])
            if len(id[i]) > 3:
                id[i] = id[i][:-4]
                disease_family.append(id[i])
                fullID.append(id[i])
                if len(id[i]) > 3:
                    id[i] = id[i][:-4]
                    disease_family.append(id[i])
                    fullID.append(id[i])
                    if len(id[i]) > 3:
                        id[i] = id[i][:-4]
                        disease_family.append(id[i])
                        fullID.append(id[i])
                        if len(id[i]) > 3:
                            id[i] = id[i][:-4]
                            disease_family.append(id[i])
                            fullID.append(id[i])
                            if len(id[i]) > 3:
                                id[i] = id[i][:-4]
                                disease_family.append(id[i])
                                fullID.append(id[i])
                                if len(id[i]) > 3:
                                    id[i] = id[i][:-4]
                                    disease_family.append(id[i])
                                    fullID.append(id[i])

    disease_list.append(disease_family)


id = meshid['ID'].tolist()
disease_dv = {}
countdis = len(disease)
for key in fullID:
    disease_dv[key] = round(math.log((disease_dv.get(key, 0) + 1)/countdis, 10)*(-1), 5)
id = meshid['ID'].tolist()
disease = meshid['disease'].tolist()

for i in range(len(disease)):
    disease[i] = {}

for i in range(len(disease)):

    if len(id[i]) > 3:
        disease[i][id[i]] = disease_dv[id[i]]
        id[i] = id[i][:-4]
        # print(disease[i])
        if len(id[i]) > 3:
            disease[i][id[i]] = disease_dv[id[i]]
            id[i] = id[i][:-4]
            # print(disease[i])
            if len(id[i]) > 3:
                disease[i][id[i]] = disease_dv[id[i]]
                id[i] = id[i][:-4]
                # print(disease[i])
                if len(id[i]) > 3:
                    disease[i][id[i]] = disease_dv[id[i]]
                    id[i] = id[i][:-4]
                    # print(disease[i])
                    if len(id[i]) > 3:
                        disease[i][id[i]] = disease_dv[id[i]]
                        id[i] = id[i][:-4]
                        # print(disease[i])
                        if len(id[i]) > 3:
                            disease[i][id[i]] = disease_dv[id[i]]
                            id[i] = id[i][:-4]
                            # print(disease[i])
                            if len(id[i]) > 3:
                                disease[i][id[i]] = disease_dv[id[i]]
                                id[i] = id[i][:-4]
                                # print(disease[i])
                                if len(id[i]) > 3:
                                    disease[i][id[i]] = disease_dv[id[i]]
                                    id[i] = id[i][:-4]
                                    # print(disease[i])
                                else:
                                    disease[i][id[i][:3]] = disease_dv[id[i][:3]]
                                    # print(disease[i])
                            else:
                                disease[i][id[i][:3]] = disease_dv[id[i][:3]]
                                # print(disease[i])
                        else:
                            disease[i][id[i][:3]] = disease_dv[id[i][:3]]
                            # print(disease[i])
                    else:
                        disease[i][id[i][:3]] = disease_dv[id[i][:3]]
                        # print(disease[i])
                else:
                    disease[i][id[i][:3]] = disease_dv[id[i][:3]]
                    # print(disease[i])
            else:
                disease[i][id[i][:3]] = disease_dv[id[i][:3]]
                # print(disease[i])
        else:
            disease[i][id[i][:3]] = disease_dv[id[i][:3]]
            # print(disease[i])
    else:
        disease[i][id[i][:3]] = disease_dv[id[i][:3]]
        # print(disease[i])


unique_disease = meshdis['C1'].tolist()


disease_name = meshid['disease'].tolist()
unique_disease_name = meshdis['C1'].tolist()

for i in range(len(unique_disease)):
    unique_disease[i] = {}
    for j in range(len(disease_name)):
        if unique_disease_name[i] == disease_name[j]:
            unique_disease[i].update(disease[j])


similarity = np.zeros([len(unique_disease_name), len(unique_disease_name)])


for m in range(len(unique_disease_name)):
    for n in range(len(unique_disease_name)):
        denominator = sum(unique_disease[m].values()) + sum(unique_disease[n].values())
        numerator = 0
        for k, v in unique_disease[m].items():
            if k in unique_disease[n].keys():
                numerator += v + unique_disease[n].get(k)
        similarity[m, n] = round(numerator/denominator, 5)


result = pd.DataFrame(similarity)
result.to_csv('output/disSemanticSimilarity2.csv')
