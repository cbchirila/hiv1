
# it started by the batch file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import create_directory
from .config import dbs_root
from .config import descriptors
from .config import enzymes
from .config import models
from .config import model_selections

from sklearn import metrics

splits=[0,1,2,3]

# def compute_prc_vs_th(db,db_folder,enzyme,descriptor,d,y_true,y_pred):
# 	precisions,recalls,thresholds = metrics.precision_recall_curve(y_true, y_pred)
# 	print(precisions)
# 	print(recalls)

# 	d["PRC-Thresholds"]=thresholds
# 	d["PRC-Precisions"]=precisions
# 	d["PRC-Recalls"]=recalls

# 	plt.plot(precisions,recalls,marker=".")
# 	plt.title(descriptor+" "+enzyme)
# 	plt.xlabel("Thresholds")
# 	plt.ylabel("PR Curve")
# 	plt.grid(True)
# 	plt.tight_layout()	
# 	plt.savefig(db_folder+"fig - prc-th-"+db.lower()+"_"+enzyme.lower()+"-"+descriptor.lower()+".png",bbox_inches="tight",orientation="portrait",dpi=100)
# 	plt.clf()
# 	return d

# def compute_acc_vs_th(db,db_folder,enzyme,descriptor,d,y_true,y_proba):
# 	thresholds = np.linspace(0,1,100)
# 	accuracies = []
# 	for t in thresholds:
# 		y_pred = (y_proba >= t).astype(int)
# 		acc = metrics.accuracy_score(y_true, y_pred)
# 		accuracies.append(acc)	
# 	print(accuracies)
# 	d["ACC-Thresholds"]=list(thresholds)
# 	d["Accuracies"]=accuracies

# 	plt.plot(d["ACC-Thresholds"],d["Accuracies"])
# 	plt.title(descriptor+" "+enzyme)
# 	plt.xlabel("Thresholds")
# 	plt.ylabel("ACC")
# 	plt.grid(True)
# 	plt.tight_layout()	
# 	plt.savefig(db_folder+"fig - acc-th-"+db.lower()+"_"+enzyme.lower()+"-"+descriptor.lower()+".png",bbox_inches="tight",orientation="portrait",dpi=100)
# 	plt.clf()
# 	return d

def compute_pr_auc_vs_th(db,db_folder,enzyme,descriptor,d,y_true,y_proba):
	thresholds = np.linspace(0,1,100)
	apss = []
	for t in thresholds:
		y_pred = (y_proba >= t).astype(int)
		aps = metrics.average_precision_score(y_true, y_pred)
		apss.append(aps)
		if t>0.49 and t<0.51:
			t=round(t,4)
			cm = metrics.confusion_matrix(y_true, y_pred)
			disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
			disp.plot(cmap="Blues", values_format="d")
			plt.title(f"Confusion Matrix (Threshold = {t})")
			plt.savefig(db_folder+"fig - cm-"+db.lower()+"_"+enzyme.lower()+"-"+descriptor.lower()+".png",bbox_inches="tight",orientation="portrait",dpi=100)
			plt.clf()
	print(apss)
	d["APS-Thresholds"]=list(thresholds)
	d["Average Precision Scores"]=apss

	plt.plot(d["APS-Thresholds"],d["Average Precision Scores"])
	plt.title(descriptor+" "+enzyme)
	plt.xlabel("Thresholds")
	plt.ylabel("PR-AUC")
	plt.grid(True)
	plt.tight_layout()	
	plt.savefig(db_folder+"fig - pr-auc-th-"+db.lower()+"_"+enzyme.lower()+"-"+descriptor.lower()+".png",bbox_inches="tight",orientation="portrait",dpi=100)
	plt.clf()
	return d

def compute_metrics(d,y_true,y_pred_int):
	d["precision"]=metrics.precision_score(y_true,y_pred_int)
	d["recall"]=metrics.recall_score(y_true,y_pred_int)
	d["accuracy"]=metrics.accuracy_score(y_true,y_pred_int)
	d["f1 score"]=metrics.f1_score(y_true,y_pred_int)
	
	fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred_int)
	d["auc"]=metrics.auc(fpr,tpr)
	d["auc-thresholds"]=thresholds

	d["mcc"]=metrics.matthews_corrcoef(y_true,y_pred_int)

	precision,recall,thresholds=metrics.precision_recall_curve(y_true,y_pred_int)
	d["pr-auc"]=metrics.auc(recall,precision)
	d["pr-auc-thresholds"]=thresholds

	return d

def process(db,db_folder):
	df4=pd.DataFrame()
	for enzyme in enzymes:
		for descriptor in descriptors:
			df2=pd.DataFrame()
			for model_selection in model_selections:
				for split in splits:
					print(enzyme,descriptor,model_selection,split)
					df1=pd.read_csv(db_folder+"/lh-"+db+"-"+enzyme+"-"+descriptor+"-"+model_selection+"--"+str(split)+"-prediction.csv")
					df2=pd.concat([df2,df1],ignore_index=True)

			d={"y_pred":"mean"}
			if "y_true" in df2.columns.to_list():
				d["y_true"]="first"

			# df3=df2.groupby(["enzyme","descriptor","model_selection","title"]).agg(d).reset_index()
			# df3=df3.sort_values(by=["y_pred","title"],ascending=[False,True])
			# df3.to_csv(db_folder+"/lh-"+db+"-"+enzyme+"-"+descriptor+"-prediction-g1.csv",index=False)

			# df3=df2.groupby(["enzyme","descriptor","title"]).agg(d).reset_index()
			# df3=df3.sort_values(by=["y_pred","title"],ascending=[False,True])
			# df3.to_csv(db_folder+"/lh-"+db+"-"+enzyme+"-"+descriptor+"-prediction-g2.csv",index=False)

			# df3=df2.groupby(["enzyme","title"]).agg(d).reset_index()
			# df3=df3.sort_values(by=["y_pred","title"],ascending=[False,True])
			# df3.to_csv(db_folder+"/lh-"+db+"-"+enzyme+"-"+descriptor+"-prediction-g3.csv",index=False)

			df3=df2.groupby(["title"]).agg(d).reset_index()
			df3["y_pred_int"]=df3["y_pred"].round(0)
			df3=df3.sort_values(by=["y_pred","title"],ascending=[False,True])
			df3.to_csv(db_folder+"/lh-"+db+"-"+enzyme+"-"+descriptor+"-prediction-g4.csv",index=False)

			if "y_true" in df2.columns.to_list():
				d={}
				d["enzyme"]=enzyme
				d["descriptor"]=descriptor
				d=compute_metrics(d,df3["y_true"],df3["y_pred_int"])
				#d=compute_prc_vs_th(db,db_folder,enzyme,descriptor,d,df3["y_true"],df3["y_pred"])
				#d=compute_acc_vs_th(db,db_folder,enzyme,descriptor,d,df3["y_true"],df3["y_pred"])
				d=compute_pr_auc_vs_th(db,db_folder,enzyme,descriptor,d,df3["y_true"],df3["y_pred"])
				df4=df4._append(d,ignore_index=True)
				if descriptor=="MIX":
					df5=pd.DataFrame()
					df5["APS-Thresholds"]=d["APS-Thresholds"]
					df5["Average Precision Scores"]=d["Average Precision Scores"]
					df5.to_csv(db_folder+"/curve-aps-"+db+"-mix-"+enzyme.lower()+".csv",index=False)

	df4.to_csv(db_folder+"/lh-"+db+"_statistics.csv",index=False)
	return

def mean_predictions(dbs,db_folders):
	for db,db_folder in zip(dbs,db_folders):
		process(db,db_folder)
	return
