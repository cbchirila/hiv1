import datetime
import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
import os
import pandas as pd
import socket

from obj import create_directory

class History():

    def __init__(self,data):
        self.data=data
        self.df=pd.DataFrame()
        return

    def add(self,h):
        df_h=pd.DataFrame(h)
        df_h["descriptor"]=pd.Series([self.data.descriptor for i in range(len(df_h))])
        df_h["enzyme"]=pd.Series([self.data.enzyme for i in range(len(df_h))])
        df_h["scaler"]=pd.Series([self.data.scalerName for i in range(len(df_h))])
        df_h["split"]=pd.Series([self.data.split for i in range(len(df_h))])
        df_h["sign"]=pd.Series([self.data.sign for i in range(len(df_h))])
        self.df=self.df._append(df_h)

        # plotting each training data
        self.plot(h)
        return

    def save(self):
        create_directory(self.data.root+"/history/")
        self.df.to_csv(self.data.root+"/history/"+self.data.modelName+"-acc-loss-hist.csv",index=False)
        return

    def plot(self,h):
        plt.plot(h["binary_accuracy"])
        plt.plot(h["loss"])
        plt.plot(h["val_binary_accuracy"])
        plt.plot(h["val_loss"])

        plt.title(self.data.modelName.upper()+" "+self.data.descriptor+" "+self.data.enzyme+" "+self.data.scalerName+" "+str(self.data.split))
        plt.ylabel("Accuracy/Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train Acc", "Train Loss", "Val. Acc", "Val. Loss"], loc="upper left")

        create_directory(self.data.root+"/history-plots/"+self.data.modelName+"/")
        self.save_fig(self.data.root+"/history-plots/"+
        self.data.modelName+"/"+self.data.descriptor+"-"+self.data.enzyme+"-"+self.data.modelName+"-"+self.data.scalerName+"-"+str(self.data.split)+
        "-acc-loss"+
        #datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")+"-"+socket.gethostname().lower()+
        "")
        plt.cla()
        return

    def save_fig(self,filename):
        plt.savefig(filename+".png", bbox_inches="tight", orientation="portrait", dpi=100)
        #plt.savefig(filename+".eps", bbox_inches="tight", orientation="portrait", dpi=300)
        return

    def clean(self):
        r=self.data.root
        m=self.data.modelName
        p=r+"/history/"+m+"-acc-loss-hist-ch.csv"
        if os.path.exists(p):
            os.remove(p)
        return
