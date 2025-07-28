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
        self.x=None
        self.y=None

        # 80% x,y
        self.x_work=None
        self.y_work=None

        # 60% x,y or 75% x_work,y_work
        self.x_train=None
        self.y_train=None

        # 20% x,y or 25% x_work,y_work
        self.x_validation=None
        self.y_validation=None

        # 20% x,y
        self.x_test=None
        self.y_test=None
        self.y_pred_test=None

        # for external predictions
        self.y_pred=None

        self.modelName=""
        self.descriptor=None
        self.enzyme=None
        self.scaler=None
        self.scalerName=""
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
        logging.debug("reading data "+descriptor+" "+enzyme)        
        self.df=pd.read_csv(self.root+"/data/"+self.database+"-HIV1-"+enzyme+"-"+descriptor+".csv")
        self.df.sort_values(by=["Title"],inplace=True)
        self.titles=self.df["Title"]

        if "pIC50" in self.df.columns:
            self.y=self.df[["Title","pIC50"]]
            self.df=self.df.drop(["pIC50"],axis=1)

        self.x=self.df
        return
