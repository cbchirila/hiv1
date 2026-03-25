
# reads the titles from the app database and searches for the ranks of the approved drugs 

import os
import pandas as pd
from ..models.config import create_directory

db_app="./db-approved/"
dbs=["./pred-dc/"]
models={
	"IN":["rfc","knn","mlp","gpc","cnn","svc"],
	"PR":["rfc","mlp","cnn","knn","dtc","svc"],
	"RT":["rfc","knn","mlp","svc"],
	}
descriptors=["PC","ECFP4","MIX"]
enzymes=["IN","PR","RT"]
scalers=["minmax","std"]
splits=[1,2,3,4]

###############################################################################

def processDbs():
    for db in dbs:
        processModels(db)
    return

###############################################################################

def processModels(db):
    create_directory(db+"/predictions-sum-rank-app/")

    for enzyme in enzymes:
        fileNameApp=db_app+"/hiv1-"+enzyme+"-approved.csv"
        if(not os.path.exists(fileNameApp)):
            print("missing",fileNameApp)
            return
        df_app=pd.read_csv(fileNameApp,delimiter=",",header=0,skipinitialspace=True)

        df3=pd.DataFrame()

        # for each title
        for i,row in df_app.iterrows():
            d={}
            d["enzyme"]=enzyme
            title=row["title"]
            print(title)
            d["title"]=title

            for model in models[enzyme]:
                fileName=db+"/predictions-sum/tab - 020-"+model+"-pred-sum-"+enzyme+".csv"
                if(not os.path.exists(fileName)):
                    print("missing",fileName)
                    continue
                df=pd.read_csv(fileName,delimiter=",",header=0,skipinitialspace=True)            
                df2=df[df["title"].str.lower()==title.lower()]
                if(len(df2)>0):
                    rank=df2["rank"].iloc[0]
                    d[model]=rank
            df3=df3._append(d,ignore_index=True)

        df3.to_csv(db+"/predictions-sum-rank-app/tab - pred-sum-rank-app-"+enzyme+".csv",index=False)
    return

###############################################################################

def main():
    os.system("cls")
    print("starting main")
    
    processDbs()

    print("stopping main")
    return

###############################################################################

main()

###############################################################################
