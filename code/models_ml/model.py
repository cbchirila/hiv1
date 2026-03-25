
# Machine Learning Abstract Model

import logging
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sys

from .config import create_directory
from .config import descriptors
from .config import enzymes
from .config import model_selections
from .config import scalers
from .config import splits
from .history import History
from .prediction import Prediction
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

class Model():

	def __init__(self,data,predictions):
		self.name=""
		self.data=data
		self.root=data.root
		self.prediction_t=None
		self.predictions=predictions

		self.shape=None
		self.model=None
		return

	def process_files(self,epochs=200):
		logging.info("model.process_files")
		logging.info("tensorflow "+str(tf.version.VERSION))

		logging.info("setting up the seed 7")
		seed=7
		np.random.seed(seed)
		tf.random.set_seed(seed)

		# preparing observables
		self.prediction_t=Prediction(self.data)
		self.data.history=History(self.data)

		# 3 enzymes x 3 descriptors (pc, ecfp4, mix) x 3 model_selections x 1 scalers (std) x 4 splits
		self.steps=len(enzymes)*len(descriptors)*len(model_selections)*len(scalers)*len(splits)
		self.step=0
		logging.info("iterating descriptors")
		for enzyme in enzymes:
			self.data.enzyme=enzyme

			for descriptor in descriptors:
				self.data.descriptor=descriptor

				for model_selection in model_selections:
					self.data.model_selection=model_selection
					self.data.read(enzyme,descriptor,model_selection)
					
					self.shape=self.data.compute_shape(descriptor)
					self.size=self.data.compute_size(descriptor)
					
					logging.info("cleaning up predictions and their metrics")
					self.prediction_t.data.modelName=self.name
					self.prediction_t.data.descriptor=descriptor
					self.prediction_t.clean()
					for p in self.predictions:
						p.data.modelName=self.name
						p.data.enzyme=enzyme
						p.data.descriptor=descriptor
						p.data.model_selection=model_selection
						p.data.signature=""
						p.data.split=0
						p.clean()

					logging.info("iterating scalers")
					for (scaler,scalerName) in scalers:
						self.data.scaler=scaler
						self.data.scalerName=scalerName
						self.process_file(enzyme,descriptor,model_selection,scaler,scalerName,epochs)
		return

	def process_file(self,enzyme,descriptor,model_selection,sc,sn,epochs):
		logging.info("model.process_file %s %s %s",enzyme,descriptor,sn)

		# 80 20
		logging.debug("separating data into work (train_validation) and test")
		self.data.x["work"],self.data.x["test"],self.data.y["work"],self.data.y["test"]=train_test_split(self.data.x["initial"],self.data.y["initial"],test_size=1/5,shuffle=False)
		
		dd="./data-decomposed"+self.data.suf+"/"
		create_directory(self.root+dd)
		# self.data.x["work"].to_csv(db_root+"./train-ch/"+dd+"/"+model_selection+"-"+descriptor+"-"+"x_work.csv",index=False)
		# self.data.y["work"].to_csv(db_root+"./train-ch/"+dd+"/"+model_selection+"-"+descriptor+"-"+"y_work.csv",index=False)

		self.data.x["test"].to_csv(self.root+dd+model_selection+"-"+descriptor+"-"+"x_test.csv",index=False)
		self.data.y["test"].to_csv(self.root+dd+model_selection+"-"+descriptor+"-"+"y_test.csv",index=False)
		#self.data.titles_d["test"]=self.data.x["test"]["title"]

		self.data.x["test"]=self.data.x["test"].drop(["title"],axis=1)
		self.data.y["test"]=self.data.y["test"].drop(["title"],axis=1)

		self.data.x["test"]=self.data.x["test"].to_numpy()
		self.data.y["test"]=self.data.y["test"].to_numpy()

		self.data.x["test"]=sc.fit_transform(self.data.x["test"])

		logging.debug("splitting work data in stratified k folds train and validation")
		skf = StratifiedKFold(n_splits=len(splits),shuffle=False)
		self.data.split=-1
		for fold_idx,(train_idx,validation_idx) in enumerate(skf.split(self.data.x["work"],self.data.y["work"][self.data.yt_label])):
			self.data.split+=1

			#print(fold_idx)
			#print(train_idx)
			#print(validation_idx)

			self.data.x["train"]=self.data.x["work"].iloc[train_idx]
			self.data.y["train"]=self.data.y["work"].iloc[train_idx]
			#self.data.titles_d["train"]=self.data.x["train"]["title"]

			self.data.x["validation"]=self.data.x["work"].iloc[validation_idx]
			self.data.y["validation"]=self.data.y["work"].iloc[validation_idx]
			#self.data.titles_d["validation"]=self.data.x["validation"]["title"]

			self.data.x["train"].to_csv(self.root+dd+model_selection+"-"+enzyme+"-"+descriptor+"-"+"x_work"+"-"+str(self.data.split)+"-train"+".csv",index=False)
			self.data.y["train"].to_csv(self.root+dd+model_selection+"-"+enzyme+"-"+descriptor+"-"+"y_work"+"-"+str(self.data.split)+"-train"+".csv",index=False)

			self.data.x["validation"].to_csv(self.root+dd+model_selection+"-"+enzyme+"-"+descriptor+"-"+"x_work"+"-"+str(self.data.split)+"-validation"+".csv",index=False)
			self.data.y["validation"].to_csv(self.root+dd+model_selection+"-"+enzyme+"-"+descriptor+"-"+"y_work"+"-"+str(self.data.split)+"-validation"+".csv",index=False)

			self.data.x["train"]=self.data.x["train"].drop(["title"],axis=1)
			self.data.y["train"]=self.data.y["train"].drop(["title"],axis=1)            

			self.data.x["validation"]=self.data.x["validation"].drop(["title"],axis=1)
			self.data.y["validation"]=self.data.y["validation"].drop(["title"],axis=1)

			self.data.x["train"]=self.data.x["train"].to_numpy()
			self.data.y["train"]=self.data.y["train"].to_numpy()

			self.data.x["validation"]=self.data.x["validation"].to_numpy()
			self.data.y["validation"]=self.data.y["validation"].to_numpy()

			assert not np.isnan(self.data.x["train"]).any(), "Input contains NaNs"
			assert not np.isinf(self.data.x["train"]).any(), "Input contains Infs"

			assert not np.isnan(self.data.x["validation"]).any(), "Input contains NaNs"
			assert not np.isinf(self.data.x["validation"]).any(), "Input contains Infs"

			logging.debug("scaling x train and x validation")
			self.data.x["train"]=sc.fit_transform(self.data.x["train"])
			self.data.x["validation"]=sc.fit_transform(self.data.x["validation"])

			assert not np.isnan(self.data.x["train"]).any(), "Input contains NaNs"
			assert not np.isinf(self.data.x["train"]).any(), "Input contains Infs"

			assert not np.isnan(self.data.x["validation"]).any(), "Input contains NaNs"
			assert not np.isinf(self.data.x["validation"]).any(), "Input contains Infs"

			assert not np.isnan(self.data.y["train"]).any(), "Labels contain NaNs"
			assert set(np.unique(self.data.y["train"])).issubset({0.0, 1.0}), "Labels must be 0 or 1"               

			assert not np.isnan(self.data.y["validation"]).any(), "Labels contain NaNs"
			assert set(np.unique(self.data.y["validation"])).issubset({0.0, 1.0}), "Labels must be 0 or 1"               

			self.step+=1
			logging.info("running"+
				" model "+self.name.upper()+
				" enzyme "+enzyme+
				" descriptor "+descriptor+
				" model_selection "+model_selection+
				" scaler "+sn+
				" split "+str(self.data.split)+
				" "+str(self.step)+"/"+str(self.steps))
			self.run(epochs)

			# saving predictions, history after each split
			self.prediction_t.save()
			for pred_c in self.predictions:
				pred_c.save()
			self.data.history.save()
		return

	def save_model(self):
		# saving models
		create_directory(self.root+"/models/")
		self.model.save(self.root+"/models/"+self.name+"-"+self.enzyme+"-"+self.descriptor+"-"+self.model_selection+"-"+self.sn+"-"+str(self.split)+".keras")
		#tf.saved_model.save(model,self.root+"./models/"+self.name+"-"+self.enzyme+"-"+descriptor+"-"+self.model_selection+"-"+sn+"-"+str(split))
		#model.export(self.root+"./models/"+name+"-"+descriptor+"-"+self.model_selection+"-"+sn+"-"+split)
		return

	def save_object(self):
		# saving models
		create_directory(self.root+"/models/")
		pickle.dump(self.model,open(self.root+"/models/"+self.name+"-"+self.enzyme+"-"+self.descriptor+"-"+self.model_selection+"-"+self.sn+"-"+str(self.split)+"-"+self.data.sign+".p","wb"))
		return

	def fix4ch(self,str):
		if len(str)==3:
			str="0"+str
		return str
