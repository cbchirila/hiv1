
# GaussianProcessClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
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
from sklearn.gaussian_process import  GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.model_selection import ParameterGrid

class GPC(Model):
    def __init__(self,data,predictions):
        super().__init__(data,predictions)
        self.name="gpc"
        self.data.modelName=self.name
        return

    def build(self,kernel=1.0 * RBF(1.0)):
        return GaussianProcessClassifier(kernel=kernel,random_state=7)
    
    # running the model    
    def run(self,epochs=0):
        # 3 enzymes * 3 descriptor * 2 scaler * 4 splits = 72
        # 72 * 9 = 648
        grid={
            "kernel" : 
            [
                RBF(length_scale=0.5),
                RBF(length_scale=1.0),
                RBF(length_scale=2.0),
                RationalQuadratic(length_scale=1.0, alpha=1.0),
                Matern(length_scale=1.0, nu=0.5),
                Matern(length_scale=1.0, nu=1.5),
                Matern(length_scale=1.0, nu=2.5),
                DotProduct(sigma_0=1.0),
                RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
            ],
        }
        
        # iterating parameters if any
        for p in list(ParameterGrid(grid)):
            self.data.params=p
            self.data.sign=p["kernel"]
            logging.info("params "+self.name+" "+self.data.descriptor+" "+self.data.enzyme+" "+self.data.scalerName+" "+str(self.data.split))

            #logging.debug("x_train")
            #logging.debug(self.data.x_train[:10])
            #logging.debug("y_train")
            #logging.debug(self.data.y_train[:10])

            #logging.debug("to build and fit the model")
            self.data.t1=time.time()
            self.model=self.build(p["kernel"])
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

    model=GPC(data,predictions)
    model.process_files()
    return

main()
