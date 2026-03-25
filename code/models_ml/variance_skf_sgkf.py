
# it started by the batch file

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

from .config import create_directory
from .config import dbs_root
from .config import descriptors
from .config import enzymes
from .config import models
from .config import model_selections

from sklearn import metrics
from scipy.stats import wilcoxon

splits=[0,1,2,3]
metrics2=["PRC","REC","ACC","F1-Score","MCC","PR-AUC"]

def compute_metrics(d,y_true,y_pred_int):
	d["PRC"]=metrics.precision_score(y_true,y_pred_int)
	d["REC"]=metrics.recall_score(y_true,y_pred_int)
	d["ACC"]=metrics.accuracy_score(y_true,y_pred_int)
	d["F1-Score"]=metrics.f1_score(y_true,y_pred_int)	
	fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred_int)
	d["AUC"]=metrics.auc(fpr,tpr)
	#d["AUC-Thresholds"]=thresholds
	d["MCC"]=metrics.matthews_corrcoef(y_true,y_pred_int)
	precision,recall,thresholds=metrics.precision_recall_curve(y_true,y_pred_int)
	d["PR-AUC"]=metrics.auc(recall,precision)
	#d["PR-AUC-Thresholds"]=thresholds
	return d

def process(db,db_folder):
	df4=pd.DataFrame()
	df5=pd.DataFrame()
	for enzyme in enzymes:
		for descriptor in descriptors:
			d2={}
			for model_selection in model_selections:
				for split in splits:
					print(enzyme,descriptor,model_selection,split)
					df1=pd.read_csv(db_folder+"/lh-"+db+"-"+enzyme+"-"+descriptor+"-"+model_selection+"--"+str(split)+"-prediction.csv")
					df2=df1
					df3=df1

					if "y_true" in df2.columns.to_list():
						d={}
						d["Enzyme"]=enzyme
						d["Descriptor"]=descriptor
						d["Model Selection"]=model_selection.upper()
						d["Split"]=split
						d=compute_metrics(d,df3["y_true"],df3["y_pred_int"])
						df4=df4._append(d,ignore_index=True)

				d2["Enzyme"]=enzyme
				d2["Descriptor"]=descriptor
				#print(df4)
				sum=0
				for metric in metrics2:
					metrics_values=df4[(df4["Enzyme"]==enzyme) & (df4["Descriptor"]==descriptor) & (df4["Model Selection"]==model_selection.upper())]
					#print(metrics_values)
					#d2[metric+" var "+model_selection.upper()]=metrics_values[metric].var()
					sum+=metrics_values[metric].var()
				mean=sum/len(metric)
				d2["Mean Var "+model_selection.upper()]=mean
			d2["Difference"]=d2["Mean Var SGKF"]-d2["Mean Var SKF"]
			d2["More Stable"]="SGKF" if d2["Difference"] < 0 else "SKF"
			print(d2)
			df5=df5._append(d2,ignore_index=True)

	stat,p=wilcoxon(df5['Mean Var SKF'],df5['Mean Var SGKF'],alternative='two-sided')
	print(stat,p,"Significant difference" if p<0.05 else "No significant difference")

	#plt.hist(df5["Difference"])
	sns.histplot(df5["Difference"],kde=False)
	plt.xlabel("Variance Difference (SGKF - SKF)")
	plt.ylabel("Count")
	# plt.title("Distribution of Variance Differences")	
	plt.grid(True)
	plt.tight_layout()	
	plt.savefig(db_folder+"/fig - var-"+db.lower()+"-var-analysis.png",bbox_inches="tight",orientation="portrait",dpi=600)
	plt.clf()

	df4.to_csv(db_folder+"/tab - var-"+db.lower()+"_statistics.csv",index=False)
	df5.to_csv(db_folder+"/tab - var-"+db.lower()+"-var-analysis.csv",index=False)
	df5.to_latex(db_folder+"/tab - var-"+db.lower()+"-var-analysis.tex",float_format=lambda x: f"{x:.2e}",index=False)
	return

def variance_skf_sgkf(dbs,db_folders):
	for db,db_folder in zip(dbs,db_folders):
		process(db,db_folder)
	return
