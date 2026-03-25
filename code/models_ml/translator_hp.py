
import numpy as np
import os
import pandas as pd
import sys

from .config import create_directory
from .config import dbs_root
from .config import descriptors
from .config import enzymes
from .config import models
from .config import models_plt
from .config import model_selections

# the mean is computed on all splits for each hyper parameters variant model in two steps
# it works on y_pred and not on y_pred_int, thus helping cnn and mlp
def compute_mean_p(df1):
	d={"y_pred":"mean"}
	if "y_true" in df1.columns.to_list():
		d["y_true"]="first"
	df3=df1.groupby(["signature","title"]).agg(d).reset_index()
	df3=df1.groupby(["title"]).agg(d).reset_index()
	df3["y_pred_int"]=df3["y_pred"].round(0)
	return df3

def translate(model_selection,model,input,output):
	file1=input+"/"+model_selection+"/"+model+"-pred.csv"
	print(file1)
	df1=pd.read_csv(file1)
	df1=df1.fillna({"signature":"sign"})
	for enzyme in enzymes:
		for descriptor in descriptors:
			for signature in list(set(df1["signature"])):
				df2=df1[(df1["enzyme"]==enzyme) & (df1["descriptor"]==descriptor) & (df1["scaler"]=="std") & (df1["signature"]==signature)]
				assert len(df2)!=0,"empty selected list"

				df3=compute_mean_p(df2.copy().reindex())
				df3["model"]=model
				df3["enzyme"]=enzyme
				df3["descriptor"]=descriptor
				l1=["model","enzyme","descriptor","title"]
				l2=[]
				l3=["y_pred","y_pred_int"]
				if "y_true" in df3.columns.to_list():
					l2=["y_true"]
				df3=df3[l1+l2+l3]

				folder2=output+"/"+model_selection+"/"+model+"-hp/"
				create_directory(folder2)
				file2=folder2+"/"+model+"-"+enzyme+"-"+descriptor+"-"+str(signature)+".csv"
				df3.to_csv(file2,index=False)
	return

def translate_data_hp():
	input=dbs_root+"/train-ch/predictions/"
	output=dbs_root+"/train-ch-ml/data/"
	for model_selection in model_selections:
		for model in models+models_plt:
			translate(model_selection,model,input,output)

	input=dbs_root+"/pred-dc/predictions/"
	output=dbs_root+"/pred-dc-ml/data/"
	for model_selection in model_selections:
		for model in models+models_plt:
			translate(model_selection,model,input,output)
	return

if __name__=="__main__":
	translate_data_hp()
