import datetime
import logging
import numpy as np
import os
import pandas as pd
import socket
import time

from metrics import Metrics
from obj import Obj

class Prediction(Obj):

    def __init__(self,data):
        self.data=data
        self.metrics=Metrics(self.data)
        self.df=pd.DataFrame()
        #self.stamp="-"+datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")+"-"+socket.gethostname().lower()
        self.stamp=""
        return

    def execute(self,model,data_t):
        #logging.debug("copying details from training data object to prediction data object")
        self.data.model=data_t.model
        self.data.descriptor=data_t.descriptor
        self.data.enzyme=data_t.enzyme
        self.data.sc=data_t.sc
        self.data.sn=data_t.sn
        self.data.split=data_t.split
        self.data.sign=data_t.sign
        self.data.params=data_t.params

        #logging.debug("reading the data to be fed to the model to get predictions")
        self.data.read(self.data.descriptor,self.data.enzyme)

        #logging.debug("applying the scaler to the input data to be predicted")
        self.data.x_sc=self.data.sc.fit_transform(self.data.x)

        #logging.debug("data.x_sc")
        #logging.debug(self.data.x_sc[:10])

        # if self.data.database[0]=="A":
        #     logging.debug(self.data.database)
        #     logging.debug("titles")
        #     logging.debug(self.data.titles[:5])

        #logging.debug("copying the data from x to x_test since x_test is the variable used in the prediction sequence")
        self.data.x_test=self.data.x_sc
        self.data.y_test=self.data.y

        #logging.debug("data_t.x")
        #logging.debug(data_t.x[:10])

        #logging.debug("data.x_test")
        #logging.debug(self.data.x_test[:10])
        #logging.debug(self.data.x_test.shape)

        #logging.debug("executing predictions "+self.data.descriptor+" "+self.data.sn+" "+str(self.data.split))
        self.data.t3=time.time()
        self.data.y_pred_test=model.predict(self.data.x_test)
        self.data.t4=time.time()

        self.data.x_train=data_t.x_train
        self.data.y_train=data_t.y_train
        self.data.y_pred_train=model.predict(self.data.x_train)

        predictions=pd.DataFrame()
        l=len(self.data.y_pred_test.flatten())
        predictions["title"]=pd.Series([self.data.titles[i] for i in range(l)])
        predictions["descriptor"]=pd.Series([self.data.descriptor for i in range(l)])
        predictions["enzyme"]=pd.Series([self.data.enzyme for i in range(l)])
        predictions["scaler"]=pd.Series([self.data.sn for i in range(l)])
        predictions["split"]=pd.Series([self.data.split for i in range(l)])
        predictions["signature"]=pd.Series([self.data.sign for i in range(l)])
        predictions["y_test"]=pd.Series(self.data.y_test.flatten())

        predictions["y_pred_test"]=pd.Series(self.data.y_pred_test.flatten())
        y_pred_test_int=np.round(self.data.y_pred_test,0).astype(int)
        predictions["y_pred_test_int"]=pd.Series(y_pred_test_int.flatten())
        
        self.df=pd.concat([self.df,predictions],ignore_index=True)

        # if self.data.database[0]=="A":
        #     logging.debug("length "+str(l))
        #     logging.debug(self.df.head())

        #logging.debug(self.data.y_test.flatten()[:10])
        #logging.debug(self.data.y_pred_test.flatten()[:10])
        #logging.debug(y_pred_test_int[:10])
        #logging.debug(self.metrics.met)

        #logging.debug("computing metrics comparing y_test with y_pred_test")
        if(len(self.data.y)>0):
            self.metrics.compute()
        return

    def save(self):
        #logging.debug("saving predictions")
        self.file="/predictions/"+self.data.model+"-pred"+self.stamp+".csv"
        self.create_directory(self.data.root+"/predictions/")
        self.df.to_csv(self.data.root+self.file,index=False)

        # saving metrics
        if(len(self.data.y)>0):
            self.metrics.save()
        return

    def clean(self):
        logging.debug("cleaning prediction")
        m=self.data.model
        r=self.data.root
        p=r+"/predictions/"+m+"-pred.csv"
        logging.debug(p)
        if os.path.exists(p):
            os.remove(p)

        self.metrics.clean()
        return
