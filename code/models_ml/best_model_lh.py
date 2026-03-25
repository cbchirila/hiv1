
import numpy as np
import os
import pandas as pd
import sys

from sklearn import metrics

from .config import create_directory
from .config import dbs_root
from .config import descriptors
from .config import enzymes
from .config import model_selectors

input=dbs_root+"/train-ch/predictions/"
output=dbs_root+"/train-ch/statistics/"
db="ch"
models=["abc","cnn","dtc","gnb","gpc","knn","mlp","qda","rfc","svc"]

def compute_acc(df1):
	acc=metrics.accuracy_score(df1["y_true"],df1["y_pred_int"])
	return acc

def process():
	df3=pd.DataFrame()
	for model in models:
		for model_selector in model_selectors:
			file1=input+"/"+model_selector+"/"+model+"-pred.csv"
			print(file1)
			df1=pd.read_csv(file1)
			df1=df1.fillna({"signature":"default"})
			signatures=list(set(df1["signature"]))
			print(signatures)

			for enzyme in enzymes:
				for descriptor in descriptors:
					for signature in signatures:
						df2=df1[(df1["enzyme"]==enzyme) & (df1["descriptor"]==descriptor) & (df1["signature"]==signature) & (df1["scaler"]=="std")]
						print(model,model_selector,enzyme,descriptor,signature)
						assert len(df2)!=0,"empty selected list"

						acc=compute_acc(df2.copy().reindex())
						print(model,model_selector,enzyme,descriptor,signature,acc)

						d={}
						d["model"]=model
						d["enzyme"]=enzyme
						d["descriptor"]=descriptor
						d["model_selector"]=model_selector
						d["acuracy"]=acc
						df3=df3._append(d,ignore_index=True)

	folder2=output
	create_directory(folder2)
	file2=folder2+"/010-best-model.csv"
	df3.to_csv(file2,index=False)
	return

def main():
	process()
	return

if __name__=="__main__":
	main()
