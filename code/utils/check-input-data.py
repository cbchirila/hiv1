import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

dbs=["drugcentral","chembl"]
folders=["pred-dc","train-ch"]
descriptors=["PC","ECFP4","MIX"]
enzymes=["IN","PR","RT"]

os.system("cls")
print("starting")

for i in range(len(dbs)):
    f=folders[i]
    db=dbs[i]
    for descriptor in descriptors:
        for enzyme in enzymes:
            df=pd.read_csv("./"+f+"/data/"+db+"-HIV1-"+enzyme+"-"+descriptor+".csv",delimiter=",",header=0,skipinitialspace=True)
            print(db,descriptor,enzyme,df.shape)

print("stopping")
