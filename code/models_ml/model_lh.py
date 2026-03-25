
# Logistic Head

import logging
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="4"
import tensorflow as tf
import time

from .config import create_directory
from .config import config_log
from .config import dbs_root
from .data import Data
from .explainer import Explainer
from .history import History
from .model import Model
from .prediction import Prediction
from tensorflow import keras
from keras import layers
from keras import optimizers

class LH(Model):

	def __init__(self,data,predictions):
		super().__init__(data,predictions)
		self.name="lh"
		self.data.model=self.name
		return

	def build(self):
		model = keras.Sequential([
			layers.Input(shape=self.shape),
			layers.Flatten(),
			layers.Dense(1,activation='sigmoid')
		])

		model.compile(
			loss=keras.losses.BinaryCrossentropy(),
			optimizer=optimizers.Adam(learning_rate=1e-4),
			metrics=[keras.metrics.BinaryAccuracy()]
		)

		return model

	# running the model
	def run(self,epochs):
		logging.info("model_lh.run")
		#logging.info("to transform from a single line to a line of columns")
		self.data.y["train"]=np.array([[v] for v in self.data.y["train"]])
		self.data.y["validation"]=np.array([[v] for v in self.data.y["validation"]])
		self.data.y["test"]=np.array([[v] for v in self.data.y["test"]])
		# iterating parameters if any

		# logging.debug("x_train")
		# logging.debug(self.data.x_train[:2])
		# logging.debug("y_train")
		# logging.debug(self.data.y_train[:2])

		logging.info("to build and fit the model")
		es=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
		self.data.t1=time.time()
		self.model=self.build()
		h=self.model.fit(self.data.x["train"],self.data.y["train"],batch_size=16,epochs=epochs,verbose=0,validation_data=(self.data.x["validation"],self.data.y["validation"]),callbacks=[es],shuffle=False)
		self.data.t2=time.time()

		#logging.debug("to add history")
		self.data.history.add(h.history)

		#logging.debug("to do training predictions")
		self.prediction_t.execute(self.model,self.data)
			
		#logging.debug("to do customer predictions")
		for pc in self.predictions:
			#logging.debug("prediction "+pc.data.root)
			pc.execute(self.model,self.data)

		if self.data.shap:
			logging.debug("to generate SHAP explanations")
			e=Explainer(self.data,self.model)
			e.explain()

		#self.save_model()
		return

def run_lh(models=[],hp=False,removed_model="",ys=False,shap=False,shuffled_model=""):
	os.system("cls")
	config_log("lh")
	logging.info("main")

	physical_devices=tf.config.list_physical_devices('GPU')
	if(len(physical_devices)>0):
		logging.info("running on "+str(physical_devices[0]))
		tf.config.experimental.set_memory_growth(physical_devices[0],True)
		epochs=1000
	else:
		logging.info("running on cpu / no gpu")
		epochs=200

	data=Data(dbs_root+"/train-ch-ml/","chembl",models,hp,removed_model,ys,shap,shuffled_model)
	predictions=[
        Prediction(Data(dbs_root+"/pred-dc-ml/","drugcentral",models,hp,removed_model,ys,shap,shuffled_model)),
	]

	model=LH(data,predictions)
	model.process_files(epochs)
	return
