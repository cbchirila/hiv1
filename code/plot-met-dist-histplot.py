
# plots the distributions of accuracies for each enzyme and descriptor using boxenplot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
import seaborn as sns

from models.config import create_directory

db_folders=["./train-ch"]

models=["abc","cnn","dtc","gnb","gpc","knn","mlp","qda","rfc","svc","abc-plt","dtc-plt","gnb-plt","gpc-plt","knn-plt","qda-plt","rfc-plt","svc-plt"]
model_selections=["skf","sgkf"]
descriptors=["PC","ECFP4","MIX"]
enzymes=["IN","PR","RT"]
enzymes2=["HIV-1 IN","HIV-1 PR","HIV-1 RT"]
scalers=["minmax","std"]
splits=[1,2,3,4]
metrics=["PRC","REC","ACC","F1-Score","AUC","MCC"]
categories=["A", "B", "C", "D", "E", "F"]
categories2=[
	"A ($0.9 < Acc \le 1.0$)", 
	"B ($0.8 < Acc \le 0.9$)", 
	"C ($0.7 < Acc\le 0.8$)", 
	"D ($0.6 < Acc \le 0.7$)",
	"E ($0.5 < Acc \le 0.6$)",
	"F ($0.0 < Acc \le 0.5$)",]
#phases=["train","validation","test"]

sns.reset_defaults()
colors = sns.color_palette("bright",6)
palette={
	"A" : colors[0],
	"B" : colors[1],
	"C" : colors[2],
	"D" : colors[3],
	"E" : colors[4],
	"F" : colors[5],
}

###############################################################################

def plotDbs(fig,axes,rows,cols):
	for db in db_folders:
		plotModels(fig,axes,db,rows,cols)
	return

###############################################################################

def plotModels(fig,axes,db,rows,cols):
	phase="train"
	for i,model_selection in enumerate(model_selections):

		# preparing data for 1 ms
		df=pd.DataFrame()
		for model in models:
			fileName=db+"/metrics/metrics-"+model_selection+"-"+model+".csv"
			df1=pd.read_csv(fileName,delimiter=",",header=0,skipinitialspace=True)
			df1.loc[df1["enzyme"] == "IN","enzyme"] = "HIV-1 IN"
			df1.loc[df1["enzyme"] == "PR","enzyme"] = "HIV-1 PR"
			df1.loc[df1["enzyme"] == "RT","enzyme"] = "HIV-1 RT"

			df1.rename(columns={"descriptor":"Descriptor","enzyme":"Enzyme",
					"precision":"PRC",
					"recall":"REC",
					"accuracy":"ACC",
					"f1 score":"F1-Score",
					"auc":"AUC",
					"mcc":"MCC"
					},inplace=True)
			df1=df1[["Enzyme","Descriptor","PRC","REC","ACC","F1-Score","AUC","MCC"]]
			print(df1.columns.to_list())
			df=pd.concat([df,df1],ignore_index=True)

		df22=pd.DataFrame()
		for j,metric in enumerate(metrics):

			df.loc[(df[metric] > 0.0) & (df[metric] <= 0.5),"Category"] = "F"
			df.loc[(df[metric] > 0.5) & (df[metric] <= 0.6),"Category"] = "E"
			df.loc[(df[metric] > 0.6) & (df[metric] <= 0.7),"Category"] = "D"
			df.loc[(df[metric] > 0.7) & (df[metric] <= 0.8),"Category"] = "C"
			df.loc[(df[metric] > 0.8) & (df[metric] <= 0.9),"Category"] = "B"
			df.loc[(df[metric] > 0.9) & (df[metric] <= 1.0),"Category"] = "A"

			for enzyme in enzymes2:
				for descriptor in descriptors:
					df.loc[(df["Enzyme"]==enzyme) & (df["Descriptor"]==descriptor),"Enzyme x Descriptor"]=enzyme+" x "+descriptor
			
			df2=pd.DataFrame()
			for enzyme in enzymes2:
				for descriptor in descriptors:
					for category in categories:
						df2=df2._append({
							"Model":model,
							"Enzyme":enzyme,
							"Descriptor":descriptor,
							"Category":category,
							"Count-"+metric:len(df[(df["Enzyme"]==enzyme) & (df["Descriptor"]==descriptor) & (df["Category"]==category)]),
						},ignore_index=True)

			print(phase,model_selection,len(df),i,j)
			plot(axes,model_selection,metric,df,i,j)

			filename=db+"/metrics-plots/metrics-distribution-hp/"
			create_directory(filename)
			filename1=filename+"/tab - plot-met-dist-110-"+model_selection+"-"+metric.lower()
			df.to_csv(filename1+".csv",index=False)

			if not "Model" in df22.columns.to_list():
				df22=pd.concat([df22,df2],axis=1)
			else:
				df22=pd.concat([df22,df2["Count-"+metric]],axis=1)

		filename2=filename+"/tab - plot-met-dist-120-"+model_selection
		df22.to_csv(filename2+".csv",index=False)

	legend_patches = [mpatches.Patch(color=colors[i], label=categories2[i]) for i in range(len(categories))]
	fig.legend(handles=legend_patches, title="Categories (Metrics)", loc="upper center", bbox_to_anchor=(0.5, 1.125), ncol=2)
	# Adjust layout to make space for legend
	plt.subplots_adjust(top=0.94)

	filename=db+"/metrics-plots/metrics-distribution-hp/"
	create_directory(filename)
	filename2=filename+"/fig - plot-met-dist-130-hp"
	savefig(filename2)

	if fig.legends:
		fig.legends.clear()
		
	for i in range(rows):
		for j in range(cols):
			axes[i][j].cla()

	return

###############################################################################

def plot(axes,model_selection,metric,df,i,j):
	sns.histplot(data=df,x="Enzyme x Descriptor",hue="Category",ax=axes[i][j],multiple="stack",palette=palette,legend=False,hue_order=categories)
	axes[i][j].set_title(metric+" for "+model_selection.upper())
	axes[i][j].set(ylabel=metric,xlabel="Enzyme x Descriptor")
	axes[i][j].tick_params(axis="x",labelrotation=90)
	return

###############################################################################

def savefig(filename):
	plt.savefig(filename+".png", bbox_inches="tight", orientation="portrait", dpi=600)
	#plt.savefig(filename+".eps", bbox_inches="tight", orientation="portrait", dpi=600)
	#plt.savefig(filename+".tif", bbox_inches="tight", orientation="portrait", dpi=600, format="tiff", pil_kwargs={"compression":"tiff_lzw"})
	return

###############################################################################

def main():
	os.system("cls")
	print("starting main")
	
	plt.rc('text',usetex=True)
	plt.rc('font',family='serif')

	sns.set_theme(style='whitegrid',font_scale=0.5)

	rows=2
	cols=6
	fig,axes=plt.subplots(nrows=rows,ncols=cols,figsize=(8.0,4.0),tight_layout=True,dpi=600)
	fig.tight_layout(w_pad=2,h_pad=12)

	plotDbs(fig,axes,rows,cols)

	plt.close(fig)

	print("stopping main")
	return

###############################################################################

main()

###############################################################################
