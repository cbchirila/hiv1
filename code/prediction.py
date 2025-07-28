import datetime
import logging
import numpy as np
import os
import pandas as pd
import socket
import time

from obj import create_directory

# external predictions
class Prediction():

    def __init__(self,data):
        logging.info("prediction.init")        
        self.data=data
        self.df=pd.DataFrame()
        #self.stamp="-"+datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")+"-"+socket.gethostname().lower()
        self.stamp=""
        return

    def execute(self,model,data_t):
        logging.info("prediction.execute")

        logging.debug(id(data_t))
        logging.debug(id(self.data))

        logging.debug("copying details from training data object to prediction data object")
        self.data.modelName=data_t.modelName
        self.data.descriptor=data_t.descriptor
        self.data.enzyme=data_t.enzyme
        self.data.scaler=data_t.scaler
        self.data.scalerName=data_t.scalerName
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

        logging.debug("executing predictions "+self.data.descriptor+" "+self.data.scalerName+" "+str(self.data.split))
        self.data.t3=time.time()
        self.data.y_pred=model.predict(self.data.x_pred)
        self.data.t4=time.time()

        predictions=pd.DataFrame()
        l=len(self.data.y_pred.flatten())
        predictions["title"]=self.data.titles
        predictions["descriptor"]=pd.Series([self.data.descriptor for i in range(l)])
        predictions["enzyme"]=pd.Series([self.data.enzyme for i in range(l)])
        predictions["scaler"]=pd.Series([self.data.scalerName for i in range(l)])
        predictions["split"]=pd.Series([self.data.split for i in range(l)])
        predictions["signature"]=pd.Series([self.data.sign for i in range(l)])

        predictions["y_pred"]=pd.Series(self.data.y_pred.flatten())
        y_pred_int=np.round(self.data.y_pred,0).astype(int)
        predictions["y_pred_int"]=pd.Series(y_pred_int.flatten())
        
        self.df=pd.concat([self.df,predictions],ignore_index=True)
        return

    def save(self):
        logging.info("prediction.save")
        self.file="/predictions/"+self.data.modelName+"-pred"+self.stamp+".csv"
        create_directory(self.data.root+"/predictions/")
        self.df.to_csv(self.data.root+self.file,index=False)
        return

    def clean(self):
        logging.info("prediction.clean")
        m=self.data.modelName
        r=self.data.root
        p=r+"/predictions/"+m+"-pred.csv"
        logging.debug(p)
        if os.path.exists(p):
            os.remove(p)
        return
