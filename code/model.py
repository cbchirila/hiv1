
# Machine Learning Abstract Model

import logging
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import sys

from history import History
from prediction import Prediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

class Model:

    def __init__(self,data,predictions):
        self.name=""
        self.data=data
        self.root=data.root
        self.prediction_t=None
        self.predictions=predictions

        self.shape=None
        self.model=None
        return

    def process_files(self,epochs=2):
        logging.info("tensorflow "+str(tf.version.VERSION))

        logging.info("cleaning up")
        for p in self.predictions:
            p.data.model=self.name
            p.clean()

        logging.info("setting up the seed")
        seed=7
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # preparing observables
        self.prediction_t=Prediction(self.data)
        self.data.history=History(self.data)

        # descriptor enzyme scaler split
        self.steps=3*3*2*4
        self.step=0
        #logging.info("iterating elements")
        for descriptor,shape,size in [("PC",(119,1),500),("ECFP4",(1024,1),2000),("MIX",(1143,1),2500)]:
            self.data.descriptor=descriptor
            self.shape=shape
            self.size=size
            for enzyme in ["IN","PR","RT"]:
                self.data.enzyme=enzyme
                self.data.read(descriptor,enzyme)
                for (sc,sn) in [(StandardScaler(),"std"),(MinMaxScaler(),"minmax")]:
                    self.data.sc=sc
                    self.data.sn=sn
                    self.process_file(descriptor,enzyme,sc,sn,epochs)
        return

    def process_file(self,descriptor,enzyme,sc,sn,epochs):
        logging.info("processing file "+descriptor+" "+enzyme+" "+sn)

        #logging.debug("unscaled")
        #logging.debug(self.data.x[:10])
        
        #logging.info("scaling the data")
        self.data.x_sc=sc.fit_transform(self.data.x)
        
        #logging.debug("scaled")
        #logging.debug(self.data.x[:10])

        #logging.debug("splitting data for external testing")
        self.data.x_sc,self.data.y=shuffle(self.data.x_sc,self.data.y,random_state=7)
        self.data.x_w,self.data.x_test,self.data.y_w,self.data.y_test=train_test_split(self.data.x_sc,self.data.y,test_size=1/5,shuffle=False)

        ###
        #dfxw=pd.DataFrame(self.data.x_w)
        #dfxw.columns=self.data.df.columns
        #dfxw.to_csv("./train-ch/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x_work.csv",index=False)
        ###
        #dfyw=pd.DataFrame(self.data.y_w)
        #dfyw.columns=["pIC50"]
        #dfyw.to_csv("./train-ch/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y_work.csv",index=False)
        ###
        #dfxh=pd.DataFrame(self.data.x_test)
        #dfxh.columns=self.data.df.columns
        #dfxh.to_csv("./train-ch/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x_test.csv",index=False)
        ###
        #dfyh=pd.DataFrame(self.data.y_test)
        #dfyw.columns=["pIC50"]
        #dfyh.to_csv("./train-ch/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y_test.csv",index=False)
        ###

        #logging.debug("training the data in stratified k folds")
        skf = StratifiedKFold(n_splits=4,shuffle=False)
        self.data.split=0
        # train, validation
        for train,validation in skf.split(self.data.x_w,self.data.y_w):
            self.data.split+=1

            self.data.x_train=self.data.x_w[train]
            self.data.y_train=self.data.y_w[train]
            self.data.x_validation=self.data.x_w[validation]
            self.data.y_validation=self.data.y_w[validation]

            ###
            #dfxwtr=pd.DataFrame(self.data.x_train)
            #dfxwtr.columns=self.data.df.columns
            #dfxwtr.to_csv("./train-ch/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x_work"+"-"+str(self.data.split)+"-train"+".csv",index=False)
            ###
            #dfywtr=pd.DataFrame(self.data.y_train)
            #dfywtr.columns=["pIC50"]
            #dfywtr.to_csv("./train-ch/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y_work"+"-"+str(self.data.split)+"-train"+".csv",index=False)
            ###
            #dfxwte=pd.DataFrame(self.data.x_test2)
            #dfxwte.columns=self.data.df.columns
            #dfxwte.to_csv("./train-ch/data-decomposed/"+enzyme+"-"+descriptor+"-"+"x_work"+"-"+str(self.data.split)+"-test"+".csv",index=False)
            ###
            #dfywte=pd.DataFrame(self.data.y_test2)
            #dfywte.columns=["pIC50"]
            #dfywte.to_csv("./train-ch/data-decomposed/"+enzyme+"-"+descriptor+"-"+"y_work"+"-"+str(self.data.split)+"-test"+".csv",index=False)
            ###

            self.step+=1
            logging.info("running model "+self.name+" descriptor "+descriptor+" enzyme "+enzyme+" scaler "+sn+" split "+str(self.data.split)+" "+str(self.step)+"/"+str(self.steps))
            self.run(epochs)

            # saving predictions, history after each split
            self.prediction_t.save()
            for pred_c in self.predictions:
                    pred_c.save()
            self.data.history.save()
        return

    def save_model(self):
        # saving models
        self.create_directory(self.root+"/models/")
        self.model.save(self.root+"/models/"+self.name+"-"+self.descriptor+"-"+self.enzyme+"-"+self.sn+"-"+str(self.split)+".keras")
        #tf.saved_model.save(model,self.root+"./models/"+name+"-"+descriptor+"-"+enzyme+"-"+sn+"-"+str(split))
        #model.export(self.root+"./models/"+name+"-"+descriptor+"-"+enzyme+"-"+sn+"-"+split)
        return

    def save_object(self):
        # saving models
        self.create_directory(self.root+"/models/")
        pickle.dump(self.model,open(self.root+"/models/"+self.name+"-"+self.descriptor+"-"+self.enzyme+"-"+self.sn+"-"+str(self.split)+"-"+self.data.sign+".p","wb"))
        return

    def fix4ch(self,str):
        if len(str)==3:
            str="0"+str
        return str
