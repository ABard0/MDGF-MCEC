# MDGF-MCEC
circRNA-disease relation prediction approach with multi-view dual attention embedding and cooperative ensemble learning.

# Dependency 
torch 1.7.0
torch-geometric 1.6.3
sklearn
python 3.6

# Content 
./code: main code of MDGF-MCEC (take circRNA2Disease as example in main code(MDGF-MCEC.py))
./datasets: the dataset of MDGF-MCEC, including circR2Disease,circ2Disease and circRNA2Disease datasets.  

# Usage
python MDGF-MCEC.py

## related database
The four databases used to support our work are shown below：
circR2Disease database(http://bioinfo.snnu.edu.cn/CircR2Disease/)
circ2Disease database(http://bioinformatics.zju.edu.cn/circ2disease/)
circRNA2Disease database(http://cgga.org.cn:9091/circRNADisease/)
Mesh database(https://www.nlm.nih.gov/mesh/meshhome.html)

## instruction:
Take for example the circRNA-disease asociations from the circRNA2Disease database

## step 1:
Harvest the initial raw data from database(circR2Disease... list above) online.
The initial raw data include circRNA-disease associations and Mesh data.

## step 2:
Extracting similarity matrix.
Four similarity algorithms has list in folder "datasets/dataHandler" named "circFuncSimilarity.py" "DisSemanticSimilarity1.py" "DisSemanticSimilarity2.py" and "GIPSimilarity.py"

## step 3:
We take association data in circRNA2Disease as example in main code(MDGF-MCEC.py)
Five-fold cross validation results can be demonstrate in plots by running "MDGF-MCEC.py" 
CircRNA-disease association prediction part of the model starts at line 493 of "MDGF-MCEC.py"：
 ```Python
# Prediction part
if fold == num_cross_val-1:
    high_prob = []
    high_index = []
...
 ```  
## similarity replacement interface:
Due to it cannot be ruled out that other similarity extraction algorithms still exist. We set interface function for introducing other similarity extraction algorithms in our model of "prepareData2.python".
 ```Python
createSimilarityInfo(Dataset, path, name)
'''
'Dataset' is set as default.
'path' is the relative path to specified similarity matrix
'name' is the name of specified similarity matrix
'''
... 
```
There are two caveats:

1)Our model supports only four similarity feature inputs. If there is a need to add a new similarity matrix, please repleace one old matrix.

2)Similarity matrix need to satisfy the uniformed format.

<br/>
 If you have any suggestions or questions, please email me at 6201613055@stu.jiangnan.edu.cn.
