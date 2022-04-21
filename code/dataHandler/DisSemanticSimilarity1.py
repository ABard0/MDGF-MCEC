import numpy as np
import pandas as pd




meshid = pd.read_csv('data/MeSHID.csv', header=0)
disease = meshid['disease'].tolist()
id = meshid['ID'].tolist()

meshdis = pd.read_csv('data/Mesh_disease.csv', header=0)
unique_disease = meshdis['C1'].tolist()

for i in range(len(disease)):
    disease[i] = {}

print("开始计算每个病的DV")

for i in range(len(disease)):

    if len(id[i]) > 3:
        disease[i][id[i]] = 1
        id[i] = id[i][:-4]
        # print(disease[i])
        if len(id[i]) > 3:
            disease[i][id[i]] = round(1 * 0.8, 5)
            id[i] = id[i][:-4]
            # print(disease[i])
            if len(id[i]) > 3:
                disease[i][id[i]] = round(1 * 0.8 * 0.8, 5)
                id[i] = id[i][:-4]
                # print(disease[i])
                if len(id[i]) > 3:
                    disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8, 5)
                    id[i] = id[i][:-4]
                    # print(disease[i])
                    if len(id[i]) > 3:
                        disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                        id[i] = id[i][:-4]
                        # print(disease[i])
                        if len(id[i]) > 3:
                            disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                            id[i] = id[i][:-4]
                            # print(disease[i])
                            if len(id[i]) > 3:
                                disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                id[i] = id[i][:-4]
                                # print(disease[i])
                                if len(id[i]) > 3:
                                    disease[i][id[i]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                    id[i] = id[i][:-4]
                                    # print(disease[i])
                                else:
                                    disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                    # print(disease[i])
                            else:
                                disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                                # print(disease[i])
                        else:
                            disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                            # print(disease[i])
                    else:
                        disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8 * 0.8, 5)
                        # print(disease[i])
                else:
                    disease[i][id[i][:3]] = round(1 * 0.8 * 0.8 * 0.8, 5)
                    # print(disease[i])
            else:
                disease[i][id[i][:3]] = round(1 * 0.8 * 0.8, 5)
                # print(disease[i])
        else:
            disease[i][id[i][:3]] = round(1 * 0.8, 5)
            # print(disease[i])
    else:
        disease[i][id[i][:3]] = 1
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
result.to_csv('output/disSemanticSimilarity1.csv')
