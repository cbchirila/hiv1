import datetime
import logging
import numpy as np
import os
import pandas as pd
import socket

from obj import Obj
from sklearn import metrics

class Metrics(Obj):

    def __init__(self,data):
        self.data=data
        self.met=None
        self.df=pd.DataFrame()
        #self.stamp="-"+datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")+"-"+socket.gethostname().lower()
        self.stamp=""
        return

    def compute(self):
        #logging.debug("computing metrics")
        self.data.y_test=self.data.y_test.flatten()
        self.data.y_pred_test=self.data.y_pred_test.flatten()
        self.data.y_pred_test=np.round(self.data.y_pred_test,0).astype(int)

        self.data.y_train=self.data.y_train.flatten()
        self.data.y_pred_train=self.data.y_pred_train.flatten()
        self.data.y_pred_train=np.round(self.data.y_pred_train,0).astype(int)

        tn_tr,fp_tr,fn_tr,tp_tr=metrics.confusion_matrix(self.data.y_train,self.data.y_pred_train).ravel()
        tn_val,fp_val,fn_val,tp_val=metrics.confusion_matrix(self.data.y_test,self.data.y_pred_test).ravel()

        acc_train=metrics.accuracy_score(self.data.y_train,self.data.y_pred_train)
        acc_val=metrics.accuracy_score(self.data.y_test,self.data.y_pred_test)
        
        pre=metrics.precision_score(self.data.y_test,self.data.y_pred_test,labels=[0,1],zero_division=0.0)
        rec=metrics.recall_score(self.data.y_test,self.data.y_pred_test,zero_division=0.0)
        f1=metrics.f1_score(self.data.y_test,self.data.y_pred_test)
        fpr,tpr,thresholds = metrics.roc_curve(self.data.y_test,self.data.y_pred_test)
        auc=metrics.auc(fpr,tpr)
        mcc=metrics.matthews_corrcoef(self.data.y_test,self.data.y_pred_test)

        ll_tr=metrics.log_loss(self.data.y_train,self.data.y_pred_train,labels=[0,1])
        ll_val=metrics.log_loss(self.data.y_test,self.data.y_pred_test,labels=[0,1])
        
        self.met={}
        self.met["model"]=self.data.model
        self.met["descriptor"]=self.data.descriptor
        self.met["enzyme"]=self.data.enzyme
        self.met["scaler"]=self.data.sn
        self.met["split"]=self.data.split

        self.met["tr_tp"]=tp_tr
        self.met["tr_fp"]=fp_tr
        self.met["tr_tn"]=tn_tr
        self.met["tr_fn"]=fn_tr

        self.met["va_tp"]=tp_val
        self.met["va_fp"]=fp_val
        self.met["va_tn"]=tn_val
        self.met["va_fn"]=fn_val

        self.met["tr_accuracy"]=acc_train
        self.met["va_accuracy"]=acc_val

        self.met["va_precision"]=pre
        self.met["va_recall"]=rec
        self.met["va_f1"]=f1
        self.met["va_auc"]=auc
        self.met["va_mcc"]=mcc

        self.met["tr_loss"]=ll_tr
        self.met["va_loss"]=ll_val
        #self.met["tr_time"]=self.data.t2-self.data.t1
        #self.met["va_prediction_time"]=self.data.t4-self.data.t3
        self.met.update(self.data.params)

        self.df=self.df._append(self.met,ignore_index=True)
        return

    def save(self):
        self.file="/metrics/"+self.data.model+"-metrics"+self.stamp+".csv"
        logging.debug("saving metrics "+self.data.root+self.file)

        self.create_directory(self.data.root+"/metrics/")
        self.df.to_csv(self.data.root+self.file,index=False)
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
