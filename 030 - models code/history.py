
import matplotlib.pyplot as plt
plt.rc('text',usetex=True)
plt.rc('font',family='serif')
import os
import pandas as pd

from obj import Obj

class History(Obj):

    def __init__(self,data):
        self.data=data
        self.df=pd.DataFrame()
        return

    def add(self,h):
        df_h=pd.DataFrame(h)
        df_h["dtype"]=pd.Series([self.data.dtype for i in range(len(df_h))])
        df_h["enzyme"]=pd.Series([self.data.enzyme for i in range(len(df_h))])
        df_h["scaler"]=pd.Series([self.data.sn for i in range(len(df_h))])
        df_h["split"]=pd.Series([self.data.split for i in range(len(df_h))])
        df_h["sign"]=pd.Series([self.data.sign for i in range(len(df_h))])
        self.df=self.df._append(df_h)

        # plotting each training data
        self.plot(h)
        return

    def save(self):
        self.create_directory(self.data.root+"/history/")
        self.df.to_csv(self.data.root+"/history/"+self.data.model+"-acc-loss-hist.csv",index=False)
        return

    def plot(self,h):
        plt.plot(h["binary_accuracy"])
        plt.plot(h["loss"])
        plt.plot(h["val_binary_accuracy"])
        plt.plot(h["val_loss"])

        plt.title(self.data.model.upper()+" "+self.data.dtype+" "+self.data.enzyme+" "+self.data.sn+" "+str(self.data.split))
        plt.ylabel("Accuracy/Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train Acc", "Train Loss", "Val. Acc", "Val. Loss"], loc="upper left")

        self.create_directory(self.data.root+"/history-plots/"+self.data.model+"/")
        self.save_fig(self.data.root+"/history-plots/"+
        self.data.model+"/"+self.data.dtype+"-"+self.data.enzyme+"-"+self.data.model+"-"+self.data.sn+"-"+str(self.data.split)+
        "-acc-loss"+
        #"-"+datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")+
        "")
        plt.cla()
        return

    def save_fig(self,filename):
        plt.savefig(filename+".png", bbox_inches="tight", orientation="portrait", dpi=100)
        plt.savefig(filename+".eps", bbox_inches="tight", orientation="portrait", dpi=300)
        return

    def clean(self):
        r=self.data.root
        m=self.data.model
        p=r+"/history/"+m+"-acc-loss-hist-ch.csv"
        if os.path.exists(p):
            os.remove(p)
        return
