import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

dbs=["drugcentral","chembl"]
folders=["./pred-dc","./train-ch"]

descriptors=["PC","ECFP4","MIX"]
enzymes=["IN","PR","RT"]
models=["abc","cnn","dtc","gnb","gpc","knn","mlp","qda","rfc","svc"]

os.system("cls")
print("starting")

print("=====predictions=====")
dfp=pd.DataFrame()
for i in range(len(dbs)):
    db=dbs[i]
    f=folders[i]
    for model in models:
        fileName="./"+f+"/predictions/"+model+"-pred"+".csv"
        if not os.path.exists(fileName):
            continue
        df=pd.read_csv(fileName,delimiter=",",header=0,skipinitialspace=True)
        print(db,model,df.shape,set(df["enzyme"]),set(df["descriptor"]))
        (lin,col)=df.shape

        for descriptor in descriptors:
            for enzyme in enzymes:
                for split in [1,2,3,4]:
                    df1=df[(df["descriptor"]==descriptor) & (df["enzyme"]==enzyme) & (df["split"]==split)]
                    print(db,model,df1.shape)
                    (lin1,col1)=df1.shape
                    dfp=dfp._append({
                        "db":db,
                        "model":model,
                        "descriptor":descriptor,
                        "enzyme":enzyme,
                        "split":split,
                        "lin":lin,
                        "col":col,
                        "lin1":lin1,
                        "col1":col1,
                    },ignore_index=True)
    dfp.to_csv(f+"/data-check/check-output-data-predictions.csv",index=False)

print("stopping")
