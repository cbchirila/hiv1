
import logging
import pandas as pd

class Data():
    def __init__(self,input,output,database):
        self.input=input
        self.output=output
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
        self.model_selection=None
        self.model_selectionName=""
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
        self.df=pd.read_csv(self.input+"/data/"+self.database+"-HIV1-"+enzyme+"-"+descriptor+".csv")
        self.df.sort_values(by=["Title"],inplace=True)

        self.titles=self.df["Title"]
        if "Smiles" in self.df.columns:
            self.smiles_list=self.df["Smiles"]
        
        if "pIC50" in self.df.columns:
            self.y=self.df[["Title","pIC50"]]
            self.df=self.df.drop(["Smiles","pIC50"],axis=1)

        self.x=self.df
        return
