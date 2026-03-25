import datetime
import logging
import numpy as np
import os
import pandas as pd
import socket
import time

from .config import phases
from .config import create_directory

# models a prediction from an external file computing no metrics
class Prediction():

	def __init__(self,data):
		logging.info("prediction.init")
		self.data=data
		self.df=pd.DataFrame()
		return

	def execute(self,model_tf,data_t):
		logging.info("prediction.execute %s",data_t.modelName.upper())
		logging.debug("data_t %s",id(data_t))
		logging.debug("self.data %s",id(self.data))
		logging.debug(data_t)
		logging.debug(self.data)

		#logging.debug("copying details from training data object to prediction data object")
		self.data.modelName=data_t.modelName
		self.data.enzyme=data_t.enzyme
		self.data.descriptor=data_t.descriptor
		self.data.model_selection=data_t.model_selection
		self.data.scaler=data_t.scaler
		self.data.scalerName=data_t.scalerName
		self.data.split=data_t.split
		self.data.signature=data_t.signature
		self.data.params=data_t.params
		self.data.removed_model=data_t.removed_model
		self.data.ys=data_t.ys
		self.data.suf=data_t.suf
		
		# reading the data to be fed to the model to get predictions
		self.data.read(self.data.enzyme,self.data.descriptor,self.data.model_selection)

		self.data.x["initial"]=self.data.x["initial"].drop(["title"],axis=1)

		# applying the scaler to the input data to be predicted
		self.data.x["initial"]=self.data.scaler.fit_transform(self.data.x["initial"])
		#logging.debug("data.x_sc %s",self.data.x_sc[:5])

		# copying the data from x_sc to x_prediction since x_prediction is the variable used in the prediction sequence
		self.data.x["prediction"]=self.data.x["initial"]

		# logging.debug("data_t.x %s",data_t.x[:5])
		# logging.debug("data.x_test %s %s",self.data.x_test.shape,self.data.x_test[:5])
		logging.debug("executing predictions "+
			self.data.enzyme+" "+
			self.data.descriptor+" "+
			self.data.model_selection+" "+
			self.data.scalerName+" "+
			str(self.data.split))

		# prediction
		self.predict(model_tf)

		self.predictions=pd.DataFrame()
		l=len(self.data.x["prediction"])
		logging.debug("l length is %s",l)
		logging.debug("titles length is %s",len(self.data.titles))
		self.predictions["model"]=pd.Series([self.data.modelName for i in range(l)])
		self.predictions["signature"]=pd.Series([self.data.signature for i in range(l)])
		self.predictions["title"]=pd.Series([self.data.titles[i] for i in range(l)])
		self.predictions["enzyme"]=pd.Series([self.data.enzyme for i in range(l)])
		self.predictions["descriptor"]=pd.Series([self.data.descriptor for i in range(l)])
		self.predictions["model_selection"]=pd.Series([self.data.model_selection for i in range(l)])
		self.predictions["scaler"]=pd.Series([self.data.scalerName for i in range(l)])
		self.predictions["split"]=pd.Series([self.data.split for i in range(l)])
		
		if id(data_t)==id(self.data):
			self.predictions["y_true"]=pd.Series(self.data.y["initial"][self.data.yt_label])
			if self.data.ys:
				logging.info("Y Scrambled for Training")
				self.predictions["y_true"]=pd.Series(self.data.y["y_scrambled"])
			else:
				logging.info("Y True for Metrics")

		self.predictions["y_pred"]=pd.Series(self.data.y_pred)
		self.predictions["y_pred_int"]=pd.Series(self.data.y_pred_int)
		
		self.df=self.predictions
		return

	def predict(self,model_tf):
		logging.info("Prediction.predict")

		logging.debug("y_pred %s",self.data.x["prediction"][:10])
		self.data.y_pred=model_tf.predict(self.data.x["prediction"]).ravel()
		logging.debug("y_pred %s",self.data.y_pred[:10])

		self.data.y_pred_int=np.round(self.data.y_pred,0).astype(int).ravel()
		logging.debug("y_pred_int %s",self.data.y_pred_int[:10])
		return

	def save(self):
		suf=self.data.suf
		self.file="/predictions"+suf+"/"+self.data.modelName+"-"+self.data.database+"-"+self.data.enzyme+"-"+self.data.descriptor+"-"+self.data.model_selection+"-"+self.data.signature+"-"+str(self.data.split)+"-prediction.csv"
		
		create_directory(self.data.root+"/predictions"+suf+"/")
		self.df.to_csv(self.data.root+self.file,index=False)
		logging.info("prediction.save "+self.data.root+self.file)
		return

	def clean(self):
		logging.info("prediction.clean")
		self.df=pd.DataFrame()

		suf=self.data.suf
		self.file="/predictions"+suf+"/"+self.data.modelName+"-"+self.data.database+"-"+self.data.enzyme+"-"+self.data.descriptor+"-"+self.data.model_selection+"-"+self.data.signature+"-"+str(self.data.split)+"-prediction.csv"
		
		logging.debug("prediction.clean %s",self.data.root+self.file)
		if os.path.exists(self.data.root+self.file):
			os.remove(self.data.root+self.file)
		return
