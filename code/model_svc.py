
# Suport Vector Classifier
# https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python

import logging
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="4"
import tensorflow as tf
import time

from data import Data
from history import History
from model import Model
from prediction import Prediction
from sklearn import svm
from sklearn.model_selection import ParameterGrid

class SVC(Model):
    def __init__(self,data,predictions):
        super().__init__(data,predictions)
        self.name="svc"
        self.data.modelName=self.name
        return

    def build(self,kernel="rbf",degree=3,C=1.0):
        return svm.SVC(kernel=kernel,degree=degree,C=C,verbose=False,random_state=7)
    
    # running the model    
    def run(self,epochs=0):
		# 3 enzymes * 3 descriptor * 2 scaler * 4 splits = 72
        # 72 * (4*1*3 + 4*1 + 4*1) = 72 * 20 = 1440
        grid={
        "kernel" : ["poly","rbf","sigmoid"],
        "degree" : [3, 4, 5],
        "C" : [0.2, 1.0, 2.0, 8.0]
        }
        
        # iterating parameters if any
        for p in list(ParameterGrid(grid)):
            self.data.params=p
            self.data.sign=self.fix4ch(str(p["C"]))+"-"+str(p["degree"])+"-"+p["kernel"]
            logging.info("params "+self.name+" "+self.data.descriptor+" "+self.data.enzyme+" "+self.data.scalerName+" "+str(self.data.split)+" "+self.data.sign)

            # logging.info("x_train")
            # logging.info(self.data.x_train.shape)
            # logging.info(self.data.x_train)
            # logging.info("y_train")
            # logging.info(self.data.y_train.shape)
            # logging.info(self.data.y_train)
            # logging.info(np.unique(self.data.y_train))

            #logging.debug("to build and fit the model")
            self.data.t1=time.time()
            self.model=self.build(p["kernel"], p["degree"], p["C"])
            self.model.fit(self.data.x_train,self.data.y_train)
            self.data.t2=time.time()

            #logging.debug("to save history")
            #h.plot()

            #logging.debug("to do training predictions")
            self.prediction_t.execute(self.model,self.data)
            
            #logging.debug("to do customer predictions")
            for pc in self.predictions:
                #logging.debug("prediction "+pc.data.root)
                pc.execute(self.model,self.data)

            #self.save_object()
        return

def main():
    os.system("cls")
    logging.info("main")

    physical_devices=tf.config.list_physical_devices('GPU')
    if(len(physical_devices)>0):
        logging.info("running on "+str(physical_devices[0]))
        tf.config.experimental.set_memory_growth(physical_devices[0],True)
    else:
        logging.info("running on cpu / no gpu")
    

    logging.info("run")
    data=Data("./train-ch/","chembl")
    predictions=[
        Prediction(Data("./pred-dc","drugcentral"))]

    model=SVC(data,predictions)
    model.process_files()
    return

main()
