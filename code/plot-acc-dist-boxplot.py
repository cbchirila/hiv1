
# plots the distributions of accuracies for each enzyme and descriptor using boxplot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
import scipy
import seaborn as sns
import statistics as st
from .models.config import create_directory

dbs=[
	"./train-ch"
	]

models=["abc","cnn","dtc","gnb","gpc","knn","mlp","qda","rfc","svc"]
descriptors=["PC","ECFP4","MIX"]
enzymes=["IN","PR","RT"]
enzymes2=["HIV-1 IN","HIV-1 PR","HIV-1 RT"]
scalers=["minmax","std"]
splits=[1,2,3,4]
metric="Accuracy"
colors=["blue","orange","green"]
phases=["train","validation","test"]

###############################################################################

def plotDbs(fig,axes,rows,cols):
	for db in dbs:
		#print(db)
		plotModels(fig,axes,db,rows,cols)
	return

###############################################################################

def plotModels(fig,axes,db,rows,cols):
	for phase in phases:
		k=0
		for model in models:
			print(phase,model)

			i=int(k/cols)
			j=k%cols

			if i>=rows:
				continue

			fileName=db+"/metrics/metrics-"+model+"-"+phase+".csv"
			df=pd.read_csv(fileName,delimiter=",",header=0,skipinitialspace=True)

			df["descriptor_int"]=0
			df.loc[df.descriptor == "PC","descriptor_int"] = 1
			df.loc[df.descriptor == "ECFP4","descriptor_int"] = 2
			df.loc[df.descriptor == "MIX","descriptor_int"] = 3
			df.loc[df.enzyme == "IN","enzyme"] = "HIV-1 IN"
			df.loc[df.enzyme == "PR","enzyme"] = "HIV-1 PR"
			df.loc[df.enzyme == "RT","enzyme"] = "HIV-1 RT"
			df.sort_values(by=["enzyme","descriptor_int"],inplace=True)

			df.rename(columns={"model":"Model","accuracy":"Accuracy","descriptor":"Descriptor","enzyme":"Enzyme"},inplace=True)
			df=df[["Model","phase","Enzyme","Descriptor","Accuracy"]]
			print(df.columns.to_list())

			print(db,model,len(df),i,j)

			plot(axes,model,metric,phase,df,i,j)

			filename=db+"/metrics-plots/accuracy-distribution-bp/"
			create_directory(filename)
			filename=filename+"/tab - plot-acc-dist-010-bp"
			print(filename)
			df.to_csv(filename+"-"+model+"-"+phase+".csv",index=False)
			
			if model in models:
				gen_stat(db,model,phase,df)
			k+=1

		legend_patches = [mpatches.Patch(color=colors[i], label=descriptors[i]) for i in range(len(descriptors))]
		fig.legend(handles=legend_patches, title="Descriptors", loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)
		plt.subplots_adjust(top=0.95)

		filename=db+"/metrics-plots/accuracy-distribution-bp/"
		create_directory(filename)
		filename2=filename+"/fig - plot-acc-dist-020-bp-2x5-"+phase
		savefig(filename2)

		for i in range(0,rows):
			for j in range(0,cols):
				axes[i][j].cla()
	return

###############################################################################

def plot(axes,model,metric,phase,df,i,j):
	sns.boxplot(data=df,y=metric,x="Enzyme",hue="Descriptor",ax=axes[i][j],legend=False,palette="bright")
	axes[i][j].set_title(metric+" for "+model.upper()+" model in "+phase)
	axes[i][j].set(ylabel=metric,xlabel="Enzyme")
	#axes[i][j].tick_params(axis="x",labelrotation=90)
	return

###############################################################################

def savefig(filename):
	plt.savefig(filename+".png", bbox_inches="tight", orientation="portrait", dpi=600)
	#plt.savefig(filename+".eps", bbox_inches="tight", orientation="portrait", dpi=600)
	#plt.savefig(filename+".tif", bbox_inches="tight", orientation="portrait", dpi=600, format="tiff", pil_kwargs={"compression":"tiff_lzw"})
	return

###############################################################################
# central tendency: mean, median
# spread: sd, range, iqr
def gen_stat(db,model,phase,df):
	df2=pd.DataFrame()
	for enzyme in enzymes2:
		for descriptor in descriptors:
			df1=df[(df["Model"]==model) & (df["phase"]==phase) & (df["Enzyme"]==enzyme) & (df["Descriptor"]==descriptor)]
			print(df1)
			count=len(df1)
			mean=round(df1["Accuracy"].mean(),4)
			min=round(df1["Accuracy"].min(),4)
			max=round(df1["Accuracy"].max(),4)
			stdev=round(st.stdev(df1["Accuracy"]),4)
			#range=round(max-min,4)
			#iqr=round(scipy.stats.iqr(df1["Accuracy"]),4)
			#median=round(df1["Accuracy"].median(),4)
			#mode=round(df1["Accuracy"].mode(),4)
			df2=df2._append({
				#"Phase":phase.capitalize(),
				#"Model":model.upper(),
				"Enzyme":enzyme,
				"Descriptor":descriptor,
				"Mean":mean,
				"Min":min,
				"Max":max,
				"Std Dev":stdev,
				#"Range":range,
				#"IQR":iqr,
				#"Median":median,
				#"Mode":mode,
				#"Count":count,
				},ignore_index=True)
			print(db,model,enzyme,descriptor,count)
	df2.to_csv(db+"/metrics-plots/accuracy-distribution-bp/tab - plot-acc-dist-030-"+model+"-"+phase+".csv",index=False)
	df2.to_latex(db+"/metrics-plots/accuracy-distribution-bp/tab - plot-acc-dist-030-"+model+"-"+phase+".tex",float_format=lambda x: "%.4f" % x,index=False)
	return

###############################################################################

def main():
	os.system("cls")
	print("starting main")
	
	plt.rc('text',usetex=True)
	plt.rc('font',family='serif')

	sns.set_theme(style='whitegrid',font_scale=0.5)

	rows=2
	cols=5
	fig,axes=plt.subplots(nrows=rows,ncols=cols,figsize=(8.1,3.6),tight_layout=True,dpi=600)
	fig.tight_layout(w_pad=2,h_pad=4)

	plotDbs(fig,axes,rows,cols)

	plt.close(fig)

	print("stopping main")
	return

###############################################################################

main()

###############################################################################
