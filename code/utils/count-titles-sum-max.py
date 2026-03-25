
# for each enzyme and each threshold selects the titles with a sum over the k% threshold

import os
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
scalers=["minmax","std"]
splits=[1,2,3,4]

###############################################################################

def processDbs():
	for db in dbs:
		processModels(db)
	return

###############################################################################

def processModels(db):
	for enzyme in enzymes:
		for model in models[enzyme]:
			fileName=db+"/predictions-sum/tab - 010-"+model+"-pred-sum.csv"
			if(not os.path.exists(fileName)):
				print("missing",fileName)
				continue
			df=pd.read_csv(fileName,delimiter=",",header=0,skipinitialspace=True)
			titles=df["title"].tolist()
			titles=set(titles)
			titles=list(titles)
			titles.sort()
			print(db,fileName,df.shape,len(titles))

			for t in [25,30,40,50,60,70,80,90]:
				max=df[enzyme].max()
				df2=df[df[enzyme]>=max*t/100.0].copy()
				df2=df2[["title",enzyme]]
				df2.sort_values(by=[enzyme],ascending=False,inplace=True)
				folder=db+f"/predictions-sum-max-{t}/"
				create_directory(folder)
				df2.to_csv(folder+"tab - "+model+"-"+enzyme+"-pred-sum-max.csv",index=False)
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
