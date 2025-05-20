import logging
import numpy as np
import pandas as pd
"""
chembl-HIV1-IN-ECFP4.csv 
drugcentral-HIV1-IN-ECFP4.csv 
ApprovedDrugs-Decoys-HIV1-IN-ECFP4.csv
database -HIV1 -enzyme -descriptor
"""
class Data():
    def __init__(self,root,database):
        self.root=root
        self.database=database
        self.df=None
        self.titles=[]
        # original input and output
        self.x=np.array([])
        self.y=np.array([])
        # scaled input tensor
        self.x_sc=np.array([])
        # 80% x,y
        self.x_w=np.array([])
        self.y_w=np.array([])
        # 60% x,y or 75% x_w,y_w
        self.x_train=np.array([])
        self.y_train=np.array([])        
        # 20% x,y or 25% x_w,y_w
        self.x_validation=np.array([])
        self.y_validation=np.array([])        
        # 20% x,y
        self.x_test=np.array([])
        self.y_test=np.array([])
        self.y_pred_test=np.array([])
        self.model=""
        self.descriptor=None
        self.enzyme=None
        self.sc=None
        self.sn=""
        self.split=-1
        self.sign=""        
        self.params={}
        self.history=None
        self.t1=0
        self.t2=0
        self.t3=0
        self.t4=0
        return

    def read(self,descriptor,enzyme):
        #logging.debug("reading data "+descriptor+" "+enzyme)        
        self.df=pd.read_csv(self.root+"/data/"+self.database+"-HIV1-"+enzyme+"-"+descriptor+".csv")
        self.titles=self.df["Title"]
        if "pIC50" in self.df.columns:
            self.y=self.df["pIC50"].to_numpy()
            self.df=self.df.drop(["pIC50"],axis=1)

        self.df=self.df.drop(["Title"],axis=1)
        self.x=self.df.to_numpy()

        #logging.debug("UNSCALED")
        #logging.debug(self.titles[:10])
        #logging.debug(self.x[:10])
        #logging.debug(self.y[:10])
        return
