# GaussianNB
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

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
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ParameterGrid

class GNB(Model):
    def __init__(self,data,predictions):
        super().__init__(data,predictions)
        self.name="gnb"
        self.data.model=self.name
        return

    def build(self,var_smoothing=1.0):
        return GaussianNB(var_smoothing=var_smoothing)
    
    # running the model    
    def run(self,epochs=0):
        # 3 enzymes * 3 descriptors * 2 scalers * 4 splits = 72
        # 72 * 4 = 288
        grid={
            "var_smoothing" : [1e-12, 1e-9, 1e-6, 1e-3],
        }
        
        # iterating parameters if any
        for p in list(ParameterGrid(grid)):
            self.data.params=p
            self.data.sign=f"{p['var_smoothing']}"
            logging.info("params "+self.name+" "+self.data.descriptor+" "+self.data.sn+" "+str(self.data.split)+" "+self.data.sign)

            #logging.debug("x_train")
            #logging.debug(self.data.x_train[:10])
            #logging.debug("y_train")
            #logging.debug(self.data.y_train[:10])

            #logging.debug("to build and fit the model")
            self.data.t1=time.time()
            self.model=self.build(p["var_smoothing"])
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

    model=GNB(data,predictions)
    model.process_files()
    return

main()
