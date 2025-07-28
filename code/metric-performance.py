
# computing performance metric for all 3 phases: train, validation, and test

import os
import pandas as pd
from sklearn import metrics
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score

dbs=["chembl",]
db_folders=["./train-ch",]
descriptors=["PC","ECFP4","MIX"]
enzymes=["IN","PR","RT"]
scalers=["minmax","std"]
models=[
	#"abc","cnn","dtc","gnb",
	"gpc","knn","mlp",
	#"qda","rfc","svc",
	]
splits=[1,2,3,4]
phases=["train","validation","test"]

def process():
	for i in range(len(dbs)):
		db=dbs[i]
		folder=db_folders[i]

		# preparing the titles data struture
		titles={}
		for enzyme in enzymes:
			titles[enzyme]={}
			for descriptor in descriptors:
				titles[enzyme][descriptor]={}
				for split in splits:
					titles[enzyme][descriptor][split]={}
					for phase in phases:
						if phase == "test":
							path=folder+"/data-decomposed/"+enzyme+"-"+descriptor+"-x_"+phase+".csv"
						else:
							path=folder+"/data-decomposed/"+enzyme+"-"+descriptor+"-x_work-"+str(split)+"-"+phase+".csv"
						df0=pd.read_csv(path,delimiter=",",header=0,skipinitialspace=True)
						titles[enzyme][descriptor][split][phase]=list(set(df0["Title"]))
		#print(titles)

		# model enzyme descriptor
		df2=pd.DataFrame()
		d={}
		for model in models:
			print(db,model)
			path=folder+"/predictions/"+model+"-pred.csv"
			df=pd.read_csv(path,delimiter=",",header=0,skipinitialspace=True)
			df=df.fillna({"signature":"sign"})
			signatures=set(df["signature"])
			signatures=list(signatures)
			signatures.sort()

			#print(df.head())

			df3=pd.DataFrame()
			d["model"]=model
			for enzyme in enzymes:
				d["enzyme"]=enzyme
				for descriptor in descriptors:
					d["descriptor"]=descriptor

					path=folder+"/data-decomposed/"+enzyme+"-"+descriptor+"-y.csv"
					df4=pd.read_csv(path,delimiter=",",header=0,skipinitialspace=True)

					for scaler in scalers:
						d["scaler"]=scaler
						for split in splits:
							d["split"]=split
							for signature in signatures:
								d["signature"]=signature
								df2=df[(df["enzyme"]==enzyme) & (df["descriptor"]==descriptor) & (df["scaler"]==scaler) & (df["split"]==split) & (df["signature"]==signature)]

								print(db,model,enzyme,descriptor,scaler,split,signature,df2.shape)

								for phase in phases:
									titles2=titles[enzyme][descriptor][split][phase]
									print(phase,len(titles2))
									df22=df2[df2["title"].isin(titles2)]
									df44=df4[df4["Title"].isin(titles2)]

									d["phase"]=phase
									
									d["precision"]=metrics.precision_score(df44["pIC50"],df22["y_pred_int"])
									d["recall"]=metrics.recall_score(df44["pIC50"],df22["y_pred_int"])
									d["accuracy"]=metrics.accuracy_score(df44["pIC50"],df22["y_pred_int"])
									d["f1 score"]=metrics.f1_score(df44["pIC50"],df22["y_pred_int"])
									
									fpr,tpr,thresholds = metrics.roc_curve(df44["pIC50"],df22["y_pred_int"])
									d["auc"]=metrics.auc(fpr,tpr)
									d["mcc"]=metrics.matthews_corrcoef(df44["pIC50"],df22["y_pred_int"])

									#d["brier score"]=metrics.brier_score_loss(df22["y_test"],df22["y_pred_test_int"])
									#d["log loss"]=metrics.log_loss(df22["y_test"],df22["y_pred_test_int"])
									#d["zero-one loss"]=metrics.zero_one_loss(df22["y_test"],df22["y_pred_test_int"])

									#d["mae"]=mean_absolute_error(df22["y_test"],df22["y_pred_test_int"])
									#d["mse"]=mean_squared_error(df22["y_test"],df22["y_pred_test_int"])
									#d["r2"]=r2_score(df22["y_test"],df22["y_pred_test_int"])

									df3=df3._append(d,ignore_index=True)
			
			for phase in phases:
				df5=df3[df3.phase==phase]
				df5.to_csv(folder+"/metrics/metrics-"+model+"-"+phase+".csv",index=False)
			df3.to_csv(folder+"/metrics/metrics-"+model+".csv",index=False)
	return

def main():
	os.system("cls")
	print("starting")
	process()
	print("stopping")
	return

main()
