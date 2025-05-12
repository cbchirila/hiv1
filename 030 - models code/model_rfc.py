
# Random Forest Classifier
# https://www.datacamp.com/tutorial/random-forests-classifier-python

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

class RFC(Model):
    def __init__(self,data,predictions):
        super().__init__(data,predictions)
        self.name="rfc"
        self.data.model=self.name
        return

    def build(self,n_estimators=100,criterion="gini"):
        return RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,verbose=0,random_state=7)
    
    # running the model    
    def run(self,epochs=0):
        # 3 enzymes * 3 descriptor * 2 scaler * 4 splits = 72
        # 72 * 2 * 5 = 72 * 10 = 720
        grid={
        "criterion" : ["gini","entropy"],
        "n_estimators" : [100,500,1000,1500,2000]
        }
        
        # iterating parameters if any
        for p in list(ParameterGrid(grid)):
            self.data.params=p
            self.data.sign=self.fix4ch(str(p["n_estimators"]))+"-"+p["criterion"]
            logging.info("params "+self.name+" "+self.data.descriptor+" "+self.data.enzyme+" "+self.data.sn+" "+str(self.data.split)+" "+self.data.sign)

            #logging.debug("x_train")
            #logging.debug(self.data.x_train[:10])
            #logging.debug("y_train")
            #logging.debug(self.data.y_train[:10])

            #logging.debug("to build and fit the model")
            self.data.t1=time.time()
            self.model=self.build(p["n_estimators"],p["criterion"])
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
        Prediction(Data("./pred-dc","drugcentral")),
        ]

    model=RFC(data,predictions)
    model.process_files()
    return

main()
