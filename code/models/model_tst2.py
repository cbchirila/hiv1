# DecisionTreeClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid

class Test2(Model):
    def __init__(self,data,predictions):
        super().__init__(data,predictions)
        self.name="tst2"
        self.data.modelName=self.name
        return

    def build(self,max_depth=5):
        return DecisionTreeClassifier(max_depth=max_depth,random_state=7)
    
    # running the model    
    def run(self,epochs=0):
        # 3 enzymes * 3 descriptors * 2 scalers * 4 splits = 72
        # 72 * 9 = 648
        grid={
            "max_depth" : [1,3,5,10,15,20,50,100,200],
        }
        
        # iterating parameters if any
        for p in list(ParameterGrid(grid)):
            self.data.params=p
            self.data.sign=f"{p['max_depth']}"
            logging.info("params "+self.name.upper()+" "+self.data.descriptor+" "+self.data.scalerName+" "+str(self.data.split)+" "+self.data.sign)

            #logging.debug("x_train")
            #logging.debug(self.data.x_train[:10])
            #logging.debug("y_train")
            #logging.debug(self.data.y_train[:10])

            #logging.debug("to build and fit the model")
            self.data.t1=time.time()
            self.model=self.build(p["max_depth"])
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

    model=Test2(data,predictions)
    model.process_files()
    return

main()
