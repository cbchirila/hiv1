
# Machine Learning Abstract Model

import logging
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sys

from history import History
from prediction import Prediction
from prediction_t import Prediction_t
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from obj import create_directory

descriptors=[("PC",(119,1),500),("ECFP4",(1024,1),2000),("MIX",(1143,1),2500)]
enzymes=["IN","PR","RT"]
scalers=[(StandardScaler(),"std"),(MinMaxScaler(),"minmax")]
splits=4

class Model:

	def __init__(self,data,predictions):
		self.name=""
		self.data=data
		self.root=data.root
		self.prediction_t=None
		self.predictions=predictions

		self.shape=None
		self.model=None
		return

	def process_files(self,epochs=2):
		logging.info("model.process_files")
		logging.info("TensorFlow %s",tf.version.VERSION)
		logging.info("Python %s",sys.version)

		logging.info("cleaning up")
		for p in self.predictions:
			p.data.modelName=self.name
			p.clean()

		logging.info("setting up the seed")
		seed=7
		np.random.seed(seed)
		tf.random.set_seed(seed)

		# preparing observables
		self.prediction_t=Prediction_t(self.data)
		self.data.history=History(self.data)

		# descriptor enzyme scaler split
		self.steps=len(descriptors)*len(enzymes)*len(scalers)*splits
		self.step=0
		logging.info("iterating elements")
		for descriptor,shape,size in descriptors:
			self.data.descriptor=descriptor
			self.shape=shape
			self.size=size
			for enzyme in enzymes:
				self.data.enzyme=enzyme
				self.data.read(descriptor,enzyme)
				for (scaler,scalerName) in scalers:
					self.data.scaler=scaler
					self.data.scalerName=scalerName
					self.process_file(descriptor,enzyme,scaler,scalerName,epochs)
		return

	def process_file(self,descriptor,enzyme,scaler,scalerName,epochs):
		logging.info("model.process_file "+descriptor+" "+enzyme+" "+scalerName)
		
		#self.data.x,self.data.y=shuffle(self.data.x,self.data.y,random_state=7)

		logging.info("splitting the data into work and test sets")
		self.data.x_work,self.data.x_test,self.data.y_work,self.data.y_test=train_test_split(self.data.x,self.data.y,test_size=1/5,shuffle=False)

		create_directory(self.root+"/data-decomposed/")
		
		self.data.x.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x.csv",index=False)
		self.data.y.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y.csv",index=False)

		self.data.x_work.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x_work.csv",index=False)
		self.data.y_work.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y_work.csv",index=False)
		
		self.data.x_test.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x_test.csv",index=False)
		self.data.y_test.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y_test.csv",index=False)

		self.data.x_pred=self.data.x.copy()
		self.data.y_true=self.data.y.copy()
		
		self.data.x_pred=self.data.x_pred.drop(["Title"],axis=1)
		self.data.y_true=self.data.y_true.drop(["Title"],axis=1)

		self.data.x_pred=self.data.x_pred.to_numpy()
		self.data.y_true=self.data.y_true.to_numpy()

		self.data.x_pred=scaler.fit_transform(self.data.x_pred)

		logging.info("training the data in stratified k folds in train and validation")
		skf = StratifiedKFold(n_splits=splits,shuffle=False)
		self.data.split=0
		# train, validation
		for fold_idx,(train_idx,validation_idx) in enumerate(skf.split(self.data.x_work,self.data.y_work["pIC50"])):
			self.data.split+=1

			self.data.x_train=self.data.x_work.iloc[train_idx]
			self.data.y_train=self.data.y_work.iloc[train_idx]

			self.data.x_validation=self.data.x_work.iloc[validation_idx]
			self.data.y_validation=self.data.y_work.iloc[validation_idx]

			self.data.x_train.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x_work"+"-"+str(self.data.split)+"-train"+".csv",index=False)
			self.data.y_train.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y_work"+"-"+str(self.data.split)+"-train"+".csv",index=False)

			self.data.x_validation.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x_work"+"-"+str(self.data.split)+"-validation"+".csv",index=False)
			self.data.y_validation.to_csv(self.root+"/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y_work"+"-"+str(self.data.split)+"-validation"+".csv",index=False)

			self.data.x_train=self.data.x_train.drop(["Title"],axis=1)
			self.data.y_train=self.data.y_train.drop(["Title"],axis=1)

			self.data.x_validation=self.data.x_validation.drop(["Title"],axis=1)
			self.data.y_validation=self.data.y_validation.drop(["Title"],axis=1)

			self.data.x_train=self.data.x_train.to_numpy()
			self.data.y_train=self.data.y_train.to_numpy()
			
			self.data.y_train=self.data.y_train.ravel()

			self.data.x_validation=self.data.x_validation.to_numpy()
			self.data.y_validation=self.data.y_validation.to_numpy()

			logging.info("scaling the data")
			self.data.x_train=scaler.fit_transform(self.data.x_train)
			self.data.x_validation=scaler.fit_transform(self.data.x_validation)

			self.step+=1
			logging.info("running model "+self.name+" descriptor "+descriptor+" enzyme "+enzyme+" scaler "+scalerName+" split "+str(self.data.split)+" "+str(self.step)+"/"+str(self.steps))
			self.run(epochs)

			# saving predictions, history after each split
			self.prediction_t.save()
			for pred_c in self.predictions:
					pred_c.save()
			self.data.history.save()
		return

	def save_model(self):
		# saving models
		self.create_directory(self.root+"/models/")
		self.model.save(self.root+"/models/"+self.name+"-"+self.descriptor+"-"+self.enzyme+"-"+self.sn+"-"+str(self.split)+".keras")
		#tf.saved_model.save(model,self.root+"./models/"+name+"-"+descriptor+"-"+enzyme+"-"+sn+"-"+str(split))
		#model.export(self.root+"./models/"+name+"-"+descriptor+"-"+enzyme+"-"+sn+"-"+split)
		return

	def save_object(self):
		# saving models
		self.create_directory(self.root+"/models/")
		pickle.dump(self.model,open(self.root+"/models/"+self.name+"-"+self.descriptor+"-"+self.enzyme+"-"+self.sn+"-"+str(self.split)+"-"+self.data.sign+".p","wb"))
		return

	def fix4ch(self,str):
		if len(str)==3:
			str="0"+str
		return str
