import pandas as pd
import numpy as np

if __name__=="__main__":
    disSeSim = pd.read_csv('../../datasets/circRNA2Disease/CircRNA2Dasease_disSSsimilarity.txt', encoding='gb18030').values
    dd_matrix = np.delete(disSeSim, 0, axis=1)
    pd_data = pd.DataFrame(dd_matrix)

    pd_data.to_csv('../../datasets/circRNA2Disease/CircRNA2Dasease_disSSsimilarity.txt', header=False, index=False, sep='\t')
    # dd_matrix = np.delete(disSeSim, 0, axis=1)
    # pd_data = pd.DataFrame(dd_matrix)
    # pd_data.to_csv('../../datasets/circ2Disease/Circ2Dasease_disSSsimilarity.txt', header=False, index=False)

    print()