
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
from .config import scalers
from .config import model_selections
from .config import n_splits
from .history import History
from .prediction import Prediction
from .prediction_t import Prediction_t
from sklearn.model_selection import train_test_split

class Model:

	def __init__(self,data,predictions):
		self.name=""
		self.data=data
		self.input=data.input
		self.output=data.output
		self.prediction_t=None
		self.predictions=predictions

		self.shape=None
		self.model=None
		return

	def process_files(self,epochs=2):
		logging.info("model.process_files")
		logging.info("TensorFlow %s",tf.version.VERSION)
		logging.info("Python %s",sys.version)

		for(model_selection1,model_selection2,model_selectionName) in model_selections:
			logging.info("model_selection "+model_selectionName)

			self.data.model_selection1=model_selection1
			self.data.model_selection2=model_selection2
			self.data.model_selectionName=model_selectionName

			# cleaning up because all the next produced data goes into a file
			logging.info("cleaning up")
			for p in self.predictions:
				p.data.model_selectionName=model_selectionName
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
			self.steps=len(descriptors)*len(enzymes)*len(scalers)*n_splits
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
						self.process_file(descriptor,enzyme,scaler,scalerName,model_selection1,model_selection2,model_selectionName,epochs)
		return

	def process_file(self,descriptor,enzyme,scaler,scalerName,model_selection1,model_selection2,model_selectionName,epochs):
		logging.info("model.process_file "+self.name.upper()+" "+enzyme+" "+descriptor+" "+model_selectionName+" "+scalerName)
		
		logging.info("splitting the data into work and test sets")
		self.data.x_work,self.data.x_test,self.data.y_work,self.data.y_test=train_test_split(self.data.x,self.data.y,test_size=1/5,shuffle=False)

		scaffolds1=self.data.x["Scaffold"].to_list()

		# this one iteration loop guarantees unique scaffolds in the work and test datasets
		for fold_idx1,(train_idx1,validation_idx1) in enumerate(model_selection1.split(self.data.x,self.data.y["pIC50"],groups=scaffolds1)):
			self.data.x_work=self.data.x.iloc[train_idx1]
			self.data.y_work=self.data.y.iloc[train_idx1]

			self.data.x_test=self.data.x.iloc[validation_idx1]
			self.data.y_test=self.data.y.iloc[validation_idx1]

			create_directory(self.input+"/data-decomposed/")
			
			self.data.x.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"x.csv",index=False)
			self.data.y.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"y.csv",index=False)

			self.data.x_work.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"x_work.csv",index=False)
			self.data.y_work.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"y_work.csv",index=False)
			
			self.data.x_test.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"x_test.csv",index=False)
			self.data.y_test.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"y_test.csv",index=False)

			self.data.x_pred=self.data.x.copy()
			self.data.y_true=self.data.y.copy()
			
			scaffolds2=self.data.x_work["Scaffold"].to_list()

			self.data.x_pred=self.data.x_pred.drop(["Title","Scaffold"],axis=1)
			self.data.y_true=self.data.y_true.drop(["Title"],axis=1)

			self.data.x_pred=self.data.x_pred.to_numpy()
			self.data.y_true=self.data.y_true.to_numpy()

			self.data.x_pred=scaler.fit_transform(self.data.x_pred)

			logging.info("training the data using "+self.name.upper()+" "+model_selectionName+" to split in train and validation")
			self.data.split=0
			
			# train, validation
			for fold_idx2,(train_idx2,validation_idx2) in enumerate(model_selection2.split(self.data.x_work,self.data.y_work["pIC50"],groups=scaffolds2)):
				self.data.split+=1

				self.data.x_train=self.data.x_work.iloc[train_idx2]
				self.data.y_train=self.data.y_work.iloc[train_idx2]

				self.data.x_validation=self.data.x_work.iloc[validation_idx2]
				self.data.y_validation=self.data.y_work.iloc[validation_idx2]

				self.data.x_train.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"x_work"+"-"+str(self.data.split)+"-train"+".csv",index=False)
				self.data.y_train.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"y_work"+"-"+str(self.data.split)+"-train"+".csv",index=False)

				self.data.x_validation.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"x_work"+"-"+str(self.data.split)+"-validation"+".csv",index=False)
				self.data.y_validation.to_csv(self.input+"/data-decomposed/"+model_selectionName+"-"+enzyme+"-"+descriptor+"-"+"y_work"+"-"+str(self.data.split)+"-validation"+".csv",index=False)

				self.data.x_train=self.data.x_train.drop(["Title","Scaffold"],axis=1)
				self.data.y_train=self.data.y_train.drop(["Title"],axis=1)

				self.data.x_validation=self.data.x_validation.drop(["Title","Scaffold"],axis=1)
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
				logging.info("running"+
					" model "+self.name.upper()+
					" descriptor "+descriptor+
					" model_selection "+model_selectionName+
					" enzyme "+enzyme+
					" scaler "+scalerName+
					" split "+str(self.data.split)+" "+str(self.step)+"/"+str(self.steps))
				self.run(epochs)

				# saving predictions, history after each split
				self.prediction_t.save()
				for pred_c in self.predictions:
						pred_c.save()
				self.data.history.save()

			# we need only one iteration from the outer loop
			break
		return

	def save_model(self):
		# saving models
		self.create_directory(self.input+"/models/")
		self.model.save(self.input+"/models/"+
			self.name+"-"+
			self.data.enzyme+"-"+
			self.data.descriptor+"-"+
			self.data.model_selectionName+"-"+
			self.data.sn+"-"+
			str(self.data.split)+
			self.data.sign+
			".keras")
		return

	def save_object(self):
		# saving models as objects
		self.create_directory(self.input+"/models/")
		pickle.dump(self.model,open(self.input+"/models/"+
			self.name+"-"+
			self.data.descriptor+"-"+
			self.data.enzyme+"-"+
			self.data.sn+"-"+
			str(self.data.split)+"-"+
			self.data.sign+
			".p","wb"))
		return

	def fix4ch(self,str):
		if len(str)==3:
			str="0"+str
		return str
