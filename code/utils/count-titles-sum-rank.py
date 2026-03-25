
# for each enzyme we compute the sum of the ranks for each title 030
# for each enzyme the sum is computed using the champion models only

import pandas as pd
from ..models.config import create_directory

dbs=["./pred-dc"]
models={
	"IN":["rfc","knn","mlp","gpc","cnn","svc"],
	"PR":["rfc","mlp","cnn","knn","dtc","svc"],
	"RT":["rfc","knn","mlp","svc"],
	}
descriptors=["PC","ECFP4","MIX"]
enzymes=["IN","PR","RT"]

def main():
    print("starting")
    create_directory(db+"/predictions-sum/")
    for db in dbs:
        for enzyme in enzymes:
            df2=pd.DataFrame()
            for model in models[enzyme]:
                print(db,enzyme,model)
                df1=pd.read_csv(db+"/predictions-sum/tab - 020-"+model+"-pred-sum-"+enzyme+".csv")
                df1["Model"]=model
                df2=pd.concat([df2,df1],ignore_index=True)
            df_sum=df2.groupby('title',as_index=False)['rank'].sum()
            df_sorted=df_sum.sort_values(by='rank',ascending=True)
            df_sorted.to_csv(db+"/predictions-sum/tab - 030-sum-rank-"+enzyme+".csv",index=False)
            #df_sorted.to_latex(db+"/predictions-sum/tab - 030-sum-rank-"+enzyme+".csv",index=False)
    print("stopping")
    return

main()
