
# KNeighborsClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

import logging
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="4"
import tensorflow as tf
import time

from .data import Data
from .history import History
from .model import Model
from .prediction import Prediction
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid

class KNN(Model):
    def __init__(self,data,predictions):
        super().__init__(data,predictions)
        self.name="knn"
        self.data.modelName=self.name
        return

    def build(self,n_neighbors=5,weights="uniform",algorithm="auto"):
        return KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm)
    
    # running the model    
    def run(self,epochs=0):
        # 3 enzymes * 3 descriptor * 2 scaler * 4 splits = 72
        # 72 * 4*2*4 = 
        grid={
            "n_neighbors" : [5,6,7,10],
            "weights" :  ["uniform","distance"],
            "algorithm" : ["auto","ball_tree","kd_tree","brute"]        
        }
        
        # iterating parameters if any
        for p in list(ParameterGrid(grid)):
            self.data.params=p
            self.data.sign=self.fix4ch(str(p["n_neighbors"]))+"-"+p["weights"]+"-"+p["algorithm"]
            logging.info("params "+self.name.upper()+" "+self.data.descriptor+" "+self.data.enzyme+" "+self.data.scalerName+" "+str(self.data.split)+" "+self.data.sign)

            #logging.debug("x_train")
            #logging.debug(self.data.x_train[:10])
            #logging.debug("y_train")
            #logging.debug(self.data.y_train[:10])

            #logging.debug("to build and fit the model")
            self.data.t1=time.time()
            self.model=self.build(p["n_neighbors"],p["weights"],p["algorithm"])
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
    data=Data("./train-ch/","./train-ch/","chembl")
    predictions=[
        Prediction(Data("./pred-dc","./pred-dc","drugcentral")),
        ]

    model=KNN(data,predictions)
    model.process_files()
    return

main()
