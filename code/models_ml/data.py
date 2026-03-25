import logging
import numpy as np
import os
import pandas as pd

class Data():
	def __init__(self,root,database,models,hp,removed_model,ys,shap,shuffled_model):
		self.root=root
		self.database=database
		self.models=models
		self.hp=hp
		self.removed_model=removed_model
		self.ys=ys
		self.shap=shap
		self.shuffled_model=shuffled_model

		self.suf1=""
		if hp:
			self.suf1="-hp"
		
		self.suf="-"+str(len(models)).zfill(2) # -18 -10 -08
		if self.hp:
			self.suf+="-hp"
		if self.ys:
			self.suf+="-ys"

		if len(self.removed_model)>0:
			self.suf="-loo-"+self.removed_model # -loo-abc

		self.df=None
		self.titles=[]

		# initial input and output
		self.x={}
		self.y={}

		# predicted values for each phase: train, validation, test
		self.y_pred={}
		self.y_pred_int={}

		self.modelName=""
		self.descriptor=None
		self.enzyme=None
		self.scaler=None
		self.scalerName=""
		self.model_selection=None
		self.model_selectionName=""
		self.split=-1
		self.signature=""
		self.params={}
		self.history=None
		self.yt_label="y_true"
		
		self.shape=None
		self.size=None
		self.NoOfTrainedModels=-1
		return

	def compute_shape(self,descriptor):
		return (self.NoOfTrainedModels,1)

	def compute_size(self,descriptor):
		return self.NoOfTrainedModels*3
	
	def read(self,enzyme,descriptor,model_selection):
		logging.info("reading data "+enzyme+" "+descriptor+" "+model_selection)
	
		self.df=pd.DataFrame()
		
		# we concatenate vertically the predictions of each model
		# we use y_pred probabilities and not y_pred_int hard predictions
		df2=pd.DataFrame()
		for model in self.models:
			dir=self.root+"/data/"+model_selection+"/"+model+self.suf1+"/"
			for file in os.listdir(dir):
				# filtering current enzyme and descriptor files
				if not model in file or not ".csv" in file or not enzyme in file or not descriptor in file:
					continue			
				logging.info("reading data %s",dir+file)
				df1=pd.read_csv(dir+file)
				if "Title" in df1.columns:
					df1.rename(columns={"Title":"title"},inplace=True)
				df1.rename(columns={"y_pred_int":"y_pred_int_"+file[:-4]},inplace=True)
				if not "title" in df2.columns.to_list():
					df2=pd.concat([df2,df1[["title"]]],axis=1)
				if self.yt_label in df1.columns.to_list() and not self.yt_label in df2.columns.to_list():
					df2=pd.concat([df2,df1[[self.yt_label]]],axis=1)
				df2=pd.concat([df2,df1[["y_pred_int_"+file[:-4]]]],axis=1)

		if self.ys:
			logging.info("Y Scrambled Forth for Training")
			if self.yt_label in df2.columns.to_list():
				self.y["y_scrambled"]=[1-x for x in df2[self.yt_label]]
				#self.y["y_scrambled"]=df2[self.yt_label].sample(frac=1)
				df2[self.yt_label]=self.y["y_scrambled"]

		self.NoOfTrainedModels=len(df2.columns.to_list())-2
		self.df=df2
		print(self.df)

		self.titles=self.df["title"]
		self.titles_d={}
		if self.yt_label in self.df.columns:
			self.y["initial"]=self.df[["title",self.yt_label]].copy()
			self.df=self.df.drop([self.yt_label],axis=1)

		self.x["initial"]=self.df.copy()
		print(self.df)

		logging.debug(self.titles.index)
		return



# Permutation Importance

# if len(self.shuffled_model)>0:
# 	self.suf+="-pi-"+self.shuffled_model

# if len(self.removed_model)>0:
# 	logging.info("Leave One Out "+self.removed_model)

	# if model==self.removed_model:
	# 	logging.info("Leave One Out")
	# 	continue

	# if model==self.shuffled_model:
	# 	logging.info("Permutation Importance - shuffling "+model)
	# 	df2["y_pred_int_"+model]=df2["y_pred_int_"+model].sample(frac=1)

