
# Suport Vector Classifier
# https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python

import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="4"
import tensorflow as tf
import time

from .data import Data
from .model import Model
from .prediction import Prediction

class Test1(Model):
    def __init__(self,data,predictions):
        super().__init__(data,predictions)
        self.name="tst1"
        self.data.modelName=self.name
        return
    
    # running the model    
    def run(self,epochs=0):
        
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
        Prediction(Data("./pred-dc","./pred-dc","drugcentral"))]

    model=Test1(data,predictions)
    model.process_files()
    return

main()
