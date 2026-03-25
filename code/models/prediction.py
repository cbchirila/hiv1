import datetime
import logging
import numpy as np
import os
import pandas as pd
import socket
import time

from .config import create_directory

# external predictions
class Prediction():

    def __init__(self,data):
        logging.info("prediction.init")        
        self.data=data
        self.df=pd.DataFrame()
        return

    def execute(self,model,data_t):
        logging.info("prediction.execute %s %s %s %s %s",data_t.modelName.upper(),data_t.descriptor,data_t.scalerName,data_t.model_selectionName,str(data_t.split))
        
        logging.debug(id(data_t))
        logging.debug(id(self.data))

        logging.debug("copying details from training data object to prediction data object")
        self.data.modelName=data_t.modelName
        self.data.descriptor=data_t.descriptor
        self.data.enzyme=data_t.enzyme
        self.data.scaler=data_t.scaler
        self.data.scalerName=data_t.scalerName
        self.data.model_selection=data_t.model_selection
        self.data.model_selectionName=data_t.model_selectionName
        self.data.split=data_t.split
        self.data.sign=data_t.sign
        self.data.params=data_t.params

        logging.debug("reading the data to be fed to the model to get predictions")
        self.data.read(self.data.descriptor,self.data.enzyme)

        self.data.x=self.data.x.drop(["Title"],axis=1)
        self.data.x=self.data.x.to_numpy()

        logging.debug("applying the scaler to the input data to be predicted")
        self.data.x_pred=self.data.scaler.fit_transform(self.data.x)
        #logging.debug("data.x_pred %s",self.data.x_pred[:10])

        logging.debug("executing predictions "+self.data.descriptor+" "+self.data.scalerName+" "+self.data.model_selectionName+" "+str(self.data.split))
        self.data.t3=time.time()
        self.data.y_pred=model.predict(self.data.x_pred)
        self.data.t4=time.time()

        predictions=pd.DataFrame()
        l=len(self.data.y_pred.flatten())
        predictions["title"]=self.data.titles
        predictions["descriptor"]=pd.Series([self.data.descriptor for i in range(l)])
        predictions["enzyme"]=pd.Series([self.data.enzyme for i in range(l)])
        predictions["scaler"]=pd.Series([self.data.scalerName for i in range(l)])
        predictions["model_selection"]=pd.Series([self.data.model_selectionName for i in range(l)])
        predictions["split"]=pd.Series([self.data.split for i in range(l)])
        predictions["signature"]=pd.Series([self.data.sign for i in range(l)])

        predictions["y_pred"]=pd.Series(self.data.y_pred.flatten())
        y_pred_int=np.round(self.data.y_pred,0).astype(int)
        predictions["y_pred_int"]=pd.Series(y_pred_int.flatten())
        
        self.df=pd.concat([self.df,predictions],ignore_index=True)
        return

    def save(self):
        logging.info("prediction.save")
        self.file="/predictions/"+self.data.model_selectionName+"/"+self.data.modelName+"-pred.csv"
        create_directory(self.data.output+"/predictions/"+self.data.model_selectionName+"/")
        self.df.to_csv(self.data.output+self.file,index=False)
        return

    def clean(self):
        logging.info("prediction.clean")
        assert len(self.data.model_selectionName)>0,"invalid selection name"
        assert len(self.data.modelName)>0,"invalid model name"
        self.file="/predictions/"+self.data.model_selectionName+"/"+self.data.modelName+"-pred.csv"
        logging.debug(self.data.output+self.file)
        if os.path.exists(self.data.output+self.file):
            os.remove(self.data.output+self.file)
        return
