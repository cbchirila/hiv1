
# Convolutional Neural Network

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
from tensorflow import keras
from keras import layers
from keras import optimizers

class CNN(Model):

    def __init__(self,data,predictions):
        super().__init__(data,predictions)
        self.name="cnn"
        self.data.model=self.name
        return

    def build(self):
        inputs = keras.Input(shape=self.shape)

        x = layers.Conv1D(filters=6, kernel_size=7, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.MaxPooling1D(pool_size=3)(x)

        x = layers.Conv1D(filters=12, kernel_size=7, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        x = layers.MaxPooling1D(pool_size=3)(x)

        x = layers.Flatten()(x)

        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=[keras.metrics.BinaryAccuracy()]
        )
        return model

    # running the model
    def run(self,epochs):
        #logging.info("to transform from a single line to a line of columns")
        self.data.y_train=np.array([[v] for v in self.data.y_train])
        self.data.y_test=np.array([[v] for v in self.data.y_test])
        self.data.y_holdout=np.array([[v] for v in self.data.y_holdout])
        # iterating parameters if any

        #logging.debug("x_train")
        #logging.debug(self.data.x_train[:10])
        #logging.debug("y_train")
        #logging.debug(self.data.y_train[:10])

        #logging.debug("to build and fit the model")
        self.data.t1=time.time()
        self.model=self.build()
        h=self.model.fit(self.data.x_train,self.data.y_train,batch_size=16,epochs=epochs,verbose=0,validation_data=(self.data.x_test,self.data.y_test),shuffle=False)
        self.data.t2=time.time()

        #logging.debug("to add history")
        self.data.history.add(h.history)

        #logging.debug("to do training predictions")
        self.prediction_t.execute(self.model,self.data)
            
        #logging.debug("to do customer predictions")
        for pc in self.predictions:
            #logging.debug("prediction "+pc.data.root)
            pc.execute(self.model,self.data)

        #self.save_model()
        return

def main():
    os.system("cls")
    logging.info("main")

    physical_devices=tf.config.list_physical_devices('GPU')
    if(len(physical_devices)>0):
        logging.info("running on "+str(physical_devices[0]))
        tf.config.experimental.set_memory_growth(physical_devices[0],True)
        epochs=1000
    else:
        logging.info("running on cpu / no gpu")
        epochs=2

    logging.info("run")
    data=Data("./train-ch/","chembl")
    predictions=[
        Prediction(Data("./pred-add","ApprovedDrugs-Decoys")),
        Prediction(Data("./pred-ch","chembl")),
        Prediction(Data("./pred-dc","drugcentral"))]

    model=CNN(data,predictions)
    model.process_files(epochs)
    return

main()
