
# reads a prediction file and computes the sum of the y_pred_int column for each title by enzyme 010 020
# computes the frequency of each title from a prediction file by enzyme

import os
import pandas as pd
from ..models.config import create_directory

dbs=["./pred-dc"]
models=["abc","cnn","dtc","gnb","gpc","knn","mlp","qda","rfc","svc"]
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
	for model in models:
		fileName=db+"/predictions/"+model+"-pred.csv"
		if(not os.path.exists(fileName)):
			continue
		df=pd.read_csv(fileName,delimiter=",",header=0,skipinitialspace=True)
		df=df[df["descriptor"]=="MIX"]
		
		titles=df["title"].tolist()
		titles=set(titles)
		titles=list(titles)
		titles.sort()
		print(db,fileName,df.shape,len(titles))

		pivot_table = df.pivot_table(
			values="y_pred_int",
			index="title",
			columns="enzyme",
			aggfunc="sum",
			fill_value=0
    	)
		pivot_table = pivot_table.reset_index()

		folder=db+"/predictions-sum/"
		create_directory(folder)
		pivot_table.to_csv(folder+"/tab - 010-"+model+"-pred-sum.csv",index=False)
		#pivot_table.to_latex(folder+"/tab - 010-"+model+"-pred-sum.tex",index=False)

		folder=db+"/predictions-sum/"
		for enzyme in enzymes:
			df2=pivot_table[["title",enzyme]].copy()
			df2.rename(columns={enzyme:"sum"},inplace=True)
			df2.sort_values(by=["sum"],ascending=False,inplace=True,ignore_index=True)
			df2["rank"] = df2["sum"].rank(method="dense",ascending=False)

			df2.to_csv(folder+"/tab - 020-"+model+"-pred-sum-"+enzyme+".csv",index=False)
			#df2.to_latex(folder+"/tab - 020-"+model+"-pred-sum-"+enzyme+".tex",index=False)
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
