import datetime
import logging
import numpy as np
import os
import pandas as pd
import socket
import time

from obj import create_directory

# training predictions
class Prediction_t():

    def __init__(self,data):
        logging.info("prediction_t.init")
        self.data=data
        self.df=pd.DataFrame()
        self.stamp=""
        return

    def execute(self,model,data_t):
        logging.info("prediction_t.execute %s %s %s",self.data.descriptor,self.data.scalerName,str(self.data.split))
        
        logging.debug(id(data_t))
        logging.debug(id(self.data))

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
        predictions["y_true"]=pd.Series(self.data.y_true.flatten())

        predictions["y_pred"]=pd.Series(self.data.y_pred.flatten())
        y_pred_int=np.round(self.data.y_pred,0).astype(int)
        predictions["y_pred_int"]=pd.Series(y_pred_int.flatten())
        
        self.df=pd.concat([self.df,predictions],ignore_index=True)
        return

    def save(self):
        logging.info("prediction_t.save")
        self.file="/predictions/"+self.data.modelName+"-pred"+self.stamp+".csv"
        create_directory(self.data.root+"/predictions/")
        self.df.to_csv(self.data.root+self.file,index=False)
        return

    def clean(self):
        logging.info("prediction_t.clean")
        m=self.data.modelName
        r=self.data.root
        p=r+"/predictions/"+m+"-pred.csv"
        logging.debug(p)
        if os.path.exists(p):
            os.remove(p)
        return
