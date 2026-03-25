
# computing performance metric for all phases: train validation test stability

import os
import pandas as pd
import sys
from sklearn import metrics
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
from .config import dbs_root
from .config import phases
from .config import descriptors
from .config import create_directory

dbs=["ch-app","ch-app"]
db_folders=["/x1-ch-app/","/x2-ch-app/"]
models=["cnn","mlp","lh"]
scalers=["std"]
signatures=[""]
splits=[1,2,3,4]

def compute_titles(folder):
	# preparing the titles data struture from the data composition folder
	titles={}
	for descriptor in descriptors:
		titles[descriptor]={}
		for split in splits:
			titles[descriptor][split]={}
			for phase in phases:
				if phase == "test" or phase == "stability":
					path=folder+"/data-decomposed/"+descriptor+"-x_"+phase+".csv"
				else:
					path=folder+"/data-decomposed/"+descriptor+"-x_work-"+str(split)+"-"+phase+".csv"
				df0=pd.read_csv(path,delimiter=",",header=0,skipinitialspace=True)
				titles[descriptor][split][phase]=list(set(df0["title"]))
				print(descriptor,split,phase,len(titles[descriptor][split][phase]))
	
	return titles

def compute_metrics_for_model(folder,db,titles,model,descriptor,scaler,signature,split):
	print(db,model,descriptor,scaler,signature,split)

	path=folder+"/predictions/"+model+"-"+db+"-"+descriptor+"-"+signature+"-"+str(split)+"-prediction.csv"
	if not os.path.exists(path):
		print("missing",path)
		return
	
	df=pd.read_csv(path,delimiter=",",header=0,skipinitialspace=True)
	df=df.fillna({"signature":""})
	signatures=set(df["signature"])
	signatures=list(signatures)
	signatures.sort()
	#print("len signatures",len(signatures))
	print(df.head())

	d={}
	df3=pd.DataFrame()
	d["model"]=model
	d["descriptor"]=descriptor
	d["scaler"]=scaler
	d["signature"]=signature
	d["split"]=split
	df2=df

	print(db,model,descriptor,scaler,split,signature,df2.shape)

	for phase in phases:
		titles2=titles[descriptor][split][phase]
		df22=df2[df2["title"].isin(titles2)]
		assert len(df22)==len(titles2), "phase titles not found"

		#df22=df2[(df2["phase"]==phase)]
		#titles3=list(set(df22["title"]))
		#print(phase,len(titles2),len(df2),len(df21))
		#print(set(titles2).difference(set(df2["title"].to_list())))
		#print(phase,len(titles2),len(titles3))
		#print(phase,df21.shape,df22.shape)
		#sys.exit()

		d["phase"]=phase				
		d["precision"]=metrics.precision_score(df22["y_true"],df22["y_pred_int"])
		d["recall"]=metrics.recall_score(df22["y_true"],df22["y_pred_int"])
		d["accuracy"]=metrics.accuracy_score(df22["y_true"],df22["y_pred_int"])
		d["f1 score"]=metrics.f1_score(df22["y_true"],df22["y_pred_int"])
		
		fpr,tpr,thresholds = metrics.roc_curve(df22["y_true"],df22["y_pred_int"])
		d["auc"]=metrics.auc(fpr,tpr)
		d["mcc"]=metrics.matthews_corrcoef(df22["y_true"],df22["y_pred_int"])
		d["log_loss"]=metrics.log_loss(df22["y_true"],df22["y_pred_int"],labels=[0,1])

		#d["brier_score"]=metrics.brier_score_loss(df22["y_true"],df22["y_pred_int"])
		#d["zero-one_loss"]=metrics.zero_one_loss(df22["y_true"],df22["y_pred_int"])

		#d["mae"]=mean_absolute_error(df22["y_true"],df22["y_pred_int"])
		#d["mse"]=mean_squared_error(df22["y_true"],df22["y_pred_int"])
		#d["r2"]=r2_score(df22["y_true"],df22["y_pred_int"])

		df3=df3._append(d,ignore_index=True)

	return df3

def compute_metrics(folder,db,titles):
	# computing metrics for each model, descriptor, signature and split
	df2=pd.DataFrame()
	for model in models:
		for descriptor in descriptors:
			df4=pd.DataFrame()
			for signature in signatures:
				for scaler in scalers:
					for split in splits:
						df3=compute_metrics_for_model(folder,db,titles,model,descriptor,scaler,signature,split)
						df4=pd.concat([df4,df3],ignore_index=True)
			df4.to_csv(folder+"/metrics/"+model+"-"+db+"-"+descriptor+"-metrics.csv",index=False)
	return

def process():
	for i in range(len(dbs)):
		db=dbs[i]
		folder=dbs_root+db_folders[i]
		create_directory(folder+"/metrics/")

		titles=compute_titles(folder)
		compute_metrics(folder,db,titles)
	return

def main():
	os.system("cls")
	print("starting")
	process()
	print("stopping")
	return

main()
