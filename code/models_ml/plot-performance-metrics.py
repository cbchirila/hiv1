
# plot the train and validation data 
# model phase descriptor

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
import seaborn as sns

from .config import create_directory
from .config import dbs_root
from .config import descriptors
from .config import phases

dbs=["ch-app"]
folders=["/x2-ch-app/"]
models=["lh"]
metrics2=["PRC","REC","ACC","F1-Score","AUC","MCC"]

sns.reset_defaults()
colors = sns.color_palette("bright",4)

###############################################################################

def plotDbs(fig,axes,rows,cols):
    for db,folder in zip(dbs,folders):
        plotModels(fig,axes,db,folder,rows,cols)
    return

###############################################################################

def plotModels(fig,axes,db,folder,rows,cols):
    df1=pd.DataFrame()
    for model in models:
        for descriptor in descriptors:
            fileName0=dbs_root+folder+"/metrics/"+model+"-"+db+"-"+descriptor+"-metrics.csv"
            print(fileName0)
            if not os.path.exists(fileName0):
                continue
            print(model,descriptor)
            df=pd.read_csv(fileName0,delimiter=",",header=0,skipinitialspace=True)
            df1=pd.concat([df1,df])
    
    print(df1.head())
    df1.loc[df1.phase == "train","phase_int"] = 1
    df1.loc[df1.phase == "validation","phase_int"] = 2
    df1.loc[df1.phase == "test","phase_int"] = 3
    df1.loc[df1.phase == "stability","phase_int"] = 4
    df1.sort_values(by=["model","descriptor","scaler","signature","split","phase_int"],inplace=True)
    filename=dbs_root+folder+"/metrics-plots/performance-metrics/"
    create_directory(filename)
    filename1=filename+"/plot-performance-metrics-010"
    df1.to_csv(filename1+".csv",index=False)

    for model in models:
        df11=df1[df1["model"]==model]
        filename=dbs_root+folder+"/metrics-plots/performance-metrics/"
        create_directory(filename)
        filename1=filename+"/plot-performance-metrics-010"
        df11.to_csv(filename1+"-"+model+".csv",index=False)

    # stage 2 of rafinements
    # df1.loc[df1.phase == "train","phase"] = "Train"
    # df1.loc[df1.phase == "validation","phase"] = "Validation"
    # df1.loc[df1.phase == "test","phase"] = "Test"
    # df1.loc[df1.phase == "stability","phase"] = "Stability"
    df1.sort_values(by=["model","descriptor","scaler","signature","split"],inplace=True)
    df1["model"]=df1["model"].str.upper()
    df1.rename(columns={"model":"Model","phase":"Phase","descriptor":"Descriptor",
        "precision":"PRC",
        "recall":"REC",
        "accuracy":"ACC",
        "f1 score":"F1-Score",
        "auc":"AUC",
        "mcc":"MCC",
        },inplace=True)
    #print(df1.columns.to_list())
    df1.sort_values(by=["Model","Descriptor"],inplace=True)
    df1=df1[["Model","Descriptor","Phase","PRC","REC","ACC","F1-Score","AUC","MCC"]]
    filename=dbs_root+folder+"/metrics-plots/performance-metrics/"
    create_directory(filename)
    filename1=filename+"/plot-performance-metrics-015"
    df1.to_csv(filename1+".csv",index=False)

    for model in models:
        df15=df1[df1["Model"]==model.upper()]
        filename=dbs_root+folder+"/metrics-plots/performance-metrics/"
        create_directory(filename)
        filename1=filename+"/plot-performance-metrics-015"
        df15.to_csv(filename1+"-"+model+".csv",index=False)

    # mean of phase model  
    for phase in phases:
        df2=pd.DataFrame()
        for model in models:
            d={}
            #d["Phase"]=phase
            d["Model"]=model.upper()
            for metric in metrics2:
                d[metric]=df1[(df1["Model"]==model.upper()) & (df1["Phase"]==phase)][metric].mean()
            df2=df2._append(d,ignore_index=True)
        df2.sort_values(by=["ACC"],ascending=[False],inplace=True)
        filename=dbs_root+folder+"/metrics-plots/performance-metrics/"
        filename2=filename+"/plot-performance-metrics-020-pm"
        df2.to_csv(filename2+"-"+phase.lower()+".csv",index=False)
        df2.to_latex(filename2+"-"+phase.lower()+".tex",float_format=lambda x: "%.4f" % x,index=False)

    # creating the plot
    k=0
    for model in models:
        i=int(k/cols)
        j=k%cols

        if i>=rows:
            continue

        df3=pd.DataFrame()
        for phase in phases:
            for descriptor in descriptors:
                df3=df3._append({
                    "Model":model.upper(),
                    "Descriptor":descriptor,
                    "Phase":phase,
                    "ACC":df1[(df1["Model"]==model.upper()) & (df1["Descriptor"]==descriptor) & (df1["Phase"]==phase)]["ACC"].mean(),
                },ignore_index=True)
        df3.sort_values(by=["Descriptor"],ascending=[True],inplace=True)

        filename=dbs_root+folder+"/metrics-plots/performance-metrics/"
        filename3=filename+"/plot-performance-metrics-030-mdp"
        df3.to_csv(filename3+"-"+model+".csv",index=False)

        plot(axes,model,"ACC",df3,i,j)

        k+=1

    legend_patches = [mpatches.Patch(color=colors[i], label=phases[i]) for i in range(len(phases))]
    fig.legend(handles=legend_patches, title="Phases", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    plt.subplots_adjust(top=0.92)

    # plot the figure
    filename=dbs_root+folder+"/metrics-plots/performance-metrics/"
    create_directory(filename)
    filename=filename+"/plot-performance-metrics-030"
    #print(filename)
    savefig(filename)

    # clear the axes
    for i in range(0,rows):
        for j in range(0,cols):
            axes[i][j].cla()
    return

###############################################################################

def plot(axes,model,metric,df,i,j):
    sns.lineplot(data=df,x="Descriptor",y=metric,hue="Phase",ax=axes[i][j],legend=False)
    axes[i][j].set_title(metric+" for "+model.upper()+" model")
    axes[i][j].set(ylabel=metric,xlabel="Descriptor")
    axes[i][j].tick_params(axis="x",labelrotation=90)
    return

###############################################################################

def savefig(filename):
    plt.savefig(filename+".png", bbox_inches="tight", orientation="portrait", dpi=600)
    #plt.savefig(filename+".eps", bbox_inches="tight", orientation="portrait", dpi=600)
    #plt.savefig(filename+".tif", bbox_inches="tight", orientation="portrait", dpi=600, format="tiff", pil_kwargs={"compression":"tiff_lzw"})
    return

###############################################################################

def main():
    os.system("cls")
    print("starting main")
    
    plt.rc('text',usetex=True)
    plt.rc('font',family='serif')

    sns.set_theme(style='whitegrid',font_scale=0.5)

    rows=2
    cols=3
    fig,axes=plt.subplots(nrows=rows,ncols=cols,figsize=(8.1,5.8),tight_layout=True,dpi=600)
    fig.tight_layout(w_pad=2,h_pad=11)

    plotDbs(fig,axes,rows,cols)

    plt.close(fig)

    print("stopping main")
    return

###############################################################################

main()

###############################################################################
