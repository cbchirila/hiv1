
import logging
import numpy as np
import os
import pandas as pd

from obj import Obj
from sklearn import metrics

class Metrics(Obj):

    def __init__(self,data):
        self.data=data
        self.met=None
        self.df=pd.DataFrame()
        return

    def compute(self):
        #logging.debug("computing metrics")
        self.data.y_holdout=self.data.y_holdout.flatten()
        self.data.y_pred=self.data.y_pred.flatten()
        self.data.y_pred=np.round(self.data.y_pred,0).astype(int)

        tn,fp,fn,tp=metrics.confusion_matrix(self.data.y_holdout,self.data.y_pred).ravel()
        acc=metrics.accuracy_score(self.data.y_holdout,self.data.y_pred)
        pre=metrics.precision_score(self.data.y_holdout,self.data.y_pred,labels=[0,1],zero_division=0.0)
        rec=metrics.recall_score(self.data.y_holdout,self.data.y_pred,zero_division=0.0)
        f1=metrics.f1_score(self.data.y_holdout,self.data.y_pred)
        fpr,tpr,thresholds = metrics.roc_curve(self.data.y_holdout,self.data.y_pred)
        auc=metrics.auc(fpr,tpr)
        mcc=metrics.matthews_corrcoef(self.data.y_holdout,self.data.y_pred)
        ll=metrics.log_loss(self.data.y_holdout,self.data.y_pred,labels=[0,1])
        
        self.met={}
        self.met["dtype"]=self.data.dtype
        self.met["enzyme"]=self.data.enzyme
        self.met["model"]=self.data.model
        self.met["scaler"]=self.data.sn
        self.met["split"]=self.data.split
        self.met["tp"]=tp
        self.met["fp"]=fp
        self.met["tn"]=tn
        self.met["fn"]=fn
        self.met["val_accuracy"]=acc
        self.met["val_precision"]=pre
        self.met["val_recall"]=rec
        self.met["val_f1"]=f1
        self.met["auc"]=auc
        self.met["mcc"]=mcc
        self.met["val_loss"]=ll
        self.met["train_time"]=self.data.t2-self.data.t1
        self.met["prediction_time"]=self.data.t4-self.data.t3
        self.met.update(self.data.params)

        self.df=self.df._append(self.met,ignore_index=True)
        return

    def save(self):
        logging.debug("saving metrics")
        self.create_directory(self.data.root+"/metrics/")
        self.df.to_csv(self.data.root+"/metrics/"+self.data.model+"-metrics.csv",index=False)
        return

    def clean(self):
        logging.debug("cleaning metrics")
        r=self.data.root
        m=self.data.model
        p=r+"/metrics/"+m+"-metrics.csv"
        logging.debug(p)
        if os.path.exists(p):
            os.remove(p)
        return
