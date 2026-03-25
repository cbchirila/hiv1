
# plot the train, validation and test data 
# model phase enzyme descriptor

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pandas as pd
import seaborn as sns
import sys
from models.config import create_directory

dbs=["chembl"]
db_folders=["./train-ch"]
models=["abc","cnn","dtc","gnb","gpc","knn","mlp","qda","rfc","svc","abc-plt","dtc-plt","gnb-plt","gpc-plt","knn-plt","qda-plt","rfc-plt","svc-plt"]
descriptors=["PC","ECFP4","MIX"]
descriptors_int={"PC":1,"ECFP4":2,"MIX":3}
enzymes=["IN","PR","RT"]
enzymes2=["HIV-1 IN","HIV-1 PR","HIV-1 RT"]
model_selections=["skf","sgkf"]
model_selections_names=["Stratified K-Fold","Stratified Group Scafolding K-Fold"]
phases=["train","validation","test"]
phases2=["Train","Validation","Test"]
#metrics=["precision","recall","accuracy","f1 score","auc","mcc","brier score","log loss","zero-one loss","mae","mse","r2"]
#metrics=["accuracy","auc"]
metrics2=["PRC","REC","ACC","F1-Score","AUC","MCC"]

sns.reset_defaults()
colors = sns.color_palette("bright",4)

###############################################################################

def collectMetrics(db_folder):
    print("collectMetrics")
    # stage 1 collecting the metrics values from all models
    df1=pd.DataFrame()
    for model_selection in model_selections:
        for model in models:
            fileName0=db_folder+"/metrics/metrics-"+model_selection+"-"+model+".csv"
            print("reading",fileName0)
            df=pd.read_csv(fileName0,delimiter=",",header=0,skipinitialspace=True)
            df["descriptor_int"]=0
            df["phase_int"]=0
            df["model_selection"]=model_selection
            df1=pd.concat([df1,df])
    return df1

def refineColNames(db_folder,df1):
    print("refineColNames")
    df1.loc[df1.descriptor == "PC","descriptor_int"] = 1
    df1.loc[df1.descriptor == "ECFP4","descriptor_int"] = 2
    df1.loc[df1.descriptor == "MIX","descriptor_int"] = 3
    df1.loc[df1.phase == "train","phase_int"] = 1
    df1.loc[df1.phase == "validation","phase_int"] = 2
    df1.loc[df1.phase == "test","phase_int"] = 3
    df1.sort_values(by=["model","enzyme","descriptor_int","scaler","signature","split","phase_int"],inplace=True)

    # dir=db_folder+"/metrics-plots/performance-metrics/"
    # filename1=dir+"/tab - plot-performance-metrics-005"
    # df1.to_csv(filename1+".csv",index=False)

    # for model in models:
    #     df11=df1[df1["model"]==model]
    #     filename1=filename+"/tab - plot-performance-metrics-010"
    #     df11.to_csv(filename1+"-"+model+".csv",index=False)
    return df1

def refineValues(db_folder,df1):
    print("refineValues")

    # stage 2 of rafinements
    df1.loc[df1.enzyme == "IN","enzyme"] = "HIV-1 IN"
    df1.loc[df1.enzyme == "PR","enzyme"] = "HIV-1 PR"
    df1.loc[df1.enzyme == "RT","enzyme"] = "HIV-1 RT"
    df1.loc[df1.phase == "train","phase"] = "Train"
    df1.loc[df1.phase == "validation","phase"] = "Validation"
    df1.loc[df1.phase == "test","phase"] = "Test"
    df1.sort_values(by=["model","enzyme","descriptor_int","scaler","signature","split"],inplace=True)
    df1["model"]=df1["model"].str.upper()
    df1.rename(columns={
        "model_selection":"Model Selection",
        "model":"Model",
        "signature":"Signature",
        "phase":"Phase",
        "enzyme":"Enzyme",
        "descriptor":"Descriptor",
        "precision":"PRC",
        "recall":"REC",
        "accuracy":"ACC",
        "f1 score":"F1-Score",
        "auc":"AUC",
        "mcc":"MCC",
        },inplace=True)
    #print(df1.columns.to_list())
    df1.sort_values(by=["Model","Enzyme","descriptor_int"],inplace=True)
    df1=df1[["Model Selection","Model","Signature","Enzyme","Descriptor","Phase","PRC","REC","ACC","F1-Score","AUC","MCC"]]

    # dir=db_folder+"/metrics-plots/performance-metrics/"
    # filename=dir+"/tab - plot-performance-metrics-015"
    # df1.to_csv(filename+".csv",index=False)
    return df1

# def computeMeanOfModels(db_folder,df1):
#     for model in models:
#         df15=df1[df1["Model"]==model.upper()]
#         dir=db_folder+"/metrics-plots/performance-metrics/"
#         filename=dir+"/tab - plot-performance-metrics-015"+"-"+model+".csv"
#         df15.to_csv(filename,index=False)
#     return

# def computeMean1(db_folder,df1):
#     # mean of splits
#     for enzyme in enzymes2:
#         df2=pd.DataFrame()
#         for model_selection in model_selections:
#             for phase in ["Train"]:
#                 for descriptor in descriptors:
#                     for model in models:
#                         signatures=list(set(df1[df1["Model"]==model.upper()]["Signature"]))
#                         print(model,signatures)
#                         for signature in signatures:
#                             d={}
#                             #d["Phase"]=phase
#                             #d["Enzyme"]=enzyme
#                             d["Descriptor"]=descriptor.upper()
#                             d["Model"]=model.upper()
#                             d["Signature"]=signature
#                             d["Model Selection"]=model_selection.upper()
#                             for metric in metrics2:
#                                 df11=df1[(df1["Model Selection"]==model_selection) & (df1["Model"]==model.upper()) & (df1["Signature"]==signature) & 
#                                     (df1["Phase"]==phase) & (df1["Enzyme"]==enzyme) & (df1["Descriptor"]==descriptor)]
#                                 d[metric]=df11[metric].mean()
#                             df2=df2._append(d,ignore_index=True)

#         df2.sort_values(by=["ACC"],ascending=[False],inplace=True)
    
#         dir=db_folder+"/metrics-plots/performance-metrics/"
#         filename=dir+"/tab - train-enz-des-model-sign-ms-"+enzyme.lower()
#         df2.to_csv(filename+".csv",index=False)
#         df2=df2.head(30)
#         df2.to_latex(filename+".tex",float_format=lambda x: "%.4f" % x,index=False)
#     return

# def computeMean2(db_folder,df1):
#     #mean of descriptor phase enzyme model
#     for descriptor in descriptors:
#         df11=df1.copy()
#         df11=df11[df11["Descriptor"]==descriptor]
#         for phase in phases2:
#             df2=pd.DataFrame()
#             for enzyme in enzymes2:
#                 for model in models:
#                     d={}
#                     #d["Phase"]=phase
#                     d["Enzyme"]=enzyme
#                     d["Model"]=model.upper()
#                     for metric in metrics2:
#                         d[metric]=df11[(df11["Model"]==model.upper()) & (df11["Phase"]==phase) & (df11["Enzyme"]==enzyme)][metric].mean()
#                     df2=df2._append(d,ignore_index=True)
#             df2.sort_values(by=["Enzyme","ACC"],ascending=[True,False],inplace=True)
#             dir=db_folder+"/metrics-plots/performance-metrics/"
#             filename=dir+"/tab - plot-performance-metrics-025-pem"
#             df2.to_csv(filename+"-"+phase.lower()+"-"+descriptor.lower()+".csv",index=False)
#             df2.to_latex(filename+"-"+phase.lower()+"-"+descriptor.lower()+".tex",float_format=lambda x: "%.4f" % x,index=False)
#     return

# def computeMean3(db_folder,df1):
#     # mean of model phase descriptor
#     for model in models:
#         df2=pd.DataFrame()
#         for descriptor in descriptors:
#             for phase in phases2:
#                 d={}
#                 d["Model"]=model.upper()
#                 d["Descriptor"]=descriptor
#                 d["Phase"]=phase
#                 for metric in metrics2:
#                     d[metric]=df1[(df1["Model"]==model.upper()) & (df1["Phase"]==phase) & (df1["Descriptor"]==descriptor) ][metric].mean()
#                 df2=df2._append(d,ignore_index=True)
#         dir=db_folder+"/metrics-plots/performance-metrics/"
#         filename=dir+"/plot-performance-metrics-021-mpd"
#         df2.to_csv(filename+"-"+model+".csv",index=False)
#     return

###############################################################################

def plotModels1(fig,axes,db_folder,rows,cols,df1):
    print("plotModels1")

    # creating the plot
    for model_selection in model_selections:
        k=0
        for model in models:
            i=int(k/cols)
            j=k%cols
            
            df3=pd.DataFrame()
            for phase in phases2:
                for enzyme in enzymes2:
                    for descriptor in descriptors:
                        df3=df3._append({
                            "Model Selection":model_selection,
                            "Model":model.upper(),
                            "Enzyme":enzyme,
                            "Descriptor":descriptor,
                            "descriptor_int":descriptors_int[descriptor],
                            "Phase":phase,
                            "Enzyme x Descriptor":enzyme+" x "+descriptor,
                            "ACC":df1[(df1["Model Selection"]==model_selection) & (df1["Model"]==model.upper()) & (df1["Enzyme"]==enzyme) & (df1["Descriptor"]==descriptor) & (df1["Phase"]==phase)]["ACC"].mean(),
                        },ignore_index=True)
            df3.sort_values(by=["Enzyme","descriptor_int"],ascending=[True,True],inplace=True)

            dir=db_folder+"/metrics-plots/performance-metrics/"
            filename=dir+"tab - acc-ed-phase-"+model_selection+"-"+model+".csv"
            df3.to_csv(filename,index=False)

            metric="ACC"
            sns.lineplot(data=df3,x="Enzyme x Descriptor",y=metric,hue="Phase",ax=axes[i][j],legend=False)
            axes[i][j].set_title(metric+" for "+model.upper()+" model")
            axes[i][j].set(ylabel=metric,xlabel="Enzyme x Descriptor")
            axes[i][j].tick_params(axis="x",labelrotation=90)

            k+=1

        legend_patches = [mpatches.Patch(color=colors[i], label=phases[i]) for i in range(len(phases))]
        fig.legend(handles=legend_patches, title="Phases", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
        plt.subplots_adjust(top=0.92)

        # plot the figure
        dir=db_folder+"/metrics-plots/performance-metrics/"
        filename=dir+"/fig - acc-ed-phase-"+model_selection
        #print(filename)
        savefig(filename)

        # clear the axes
        for l in fig.legends:
            l.remove()
        for i in range(0,rows):
            for j in range(0,cols):
                axes[i][j].cla()
    return df1

###############################################################################

def plotModels2(fig,axes,db_folder,rows,cols,df1):
    print("plotModels2")
    # to fill the x axes for plotting
    for enzyme in enzymes2:
        for descriptor in descriptors:
            df1.loc[(df1["Enzyme"]==enzyme) & (df1["Descriptor"]==descriptor),"Enzyme x Descriptor"]=enzyme+" x "+descriptor

    # creating the plot
    k=0
    for model in models:
        i=int(k/cols)
        j=k%cols
        
        df3=pd.DataFrame()
        for model_selection in model_selections:
            for phase in ["Train"]:
                for enzyme in enzymes2:
                    for descriptor in descriptors:
                        df3=df3._append({
                            "Model Selection":model_selection,
                            "Model":model,
                            "Enzyme":enzyme,
                            "Descriptor":descriptor,
                            "descriptor_int":descriptors_int[descriptor],
                            "Phase":phase,
                            "Enzyme x Descriptor":enzyme+" x "+descriptor,
                            "ACC":df1[(df1["Model Selection"]==model_selection) & (df1["Model"]==model.upper()) & (df1["Enzyme"]==enzyme) & (df1["Descriptor"]==descriptor) & (df1["Phase"]==phase)]["ACC"].mean(),
                        },ignore_index=True)
        df3.sort_values(by=["Enzyme","descriptor_int"],ascending=[True,True],inplace=True)

        dir=db_folder+"/metrics-plots/performance-metrics/"
        filename=dir+"/tab - acctrain-ed-ms-"+model+".csv"
        df3.to_csv(filename,index=False)

        metric="ACC"
        sns.lineplot(data=df3,x="Enzyme x Descriptor",y=metric,hue="Model Selection",ax=axes[i][j],legend=False)
        axes[i][j].set_title(metric+" for "+model.upper()+" model")
        axes[i][j].set(ylabel=metric,xlabel="Enzyme x Descriptor")
        axes[i][j].tick_params(axis="x",labelrotation=90)

        k+=1

    legend_patches = [mpatches.Patch(color=colors[i], label=model_selections_names[i]) for i in range(len(model_selections_names))]
    fig.legend(handles=legend_patches, title="Model Selections", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    plt.subplots_adjust(top=0.92)

    # plot the figure
    dir=db_folder+"/metrics-plots/performance-metrics/"
    filename=dir+"/fig - acctrain-ed-ms"
    #print(filename)
    savefig(filename)

    # clear the axes
    for l in fig.legends:
        l.remove()
    for i in range(0,rows):
        for j in range(0,cols):
            axes[i][j].cla()
    return df1

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

    rows=4
    cols=5
    fig,axes=plt.subplots(nrows=rows,ncols=cols,figsize=(8.0,8.0),tight_layout=True,dpi=600)
    fig.tight_layout(w_pad=2,h_pad=13)
    fig.delaxes(axes[3][3])
    fig.delaxes(axes[3][4])

    for db_folder in db_folders:
        filename=db_folder+"/metrics-plots/performance-metrics/"
        create_directory(filename)

        df1=collectMetrics(db_folder)
        df1=refineColNames(db_folder,df1)
        df1=refineValues(db_folder,df1)

        # computeMean1(db_folder,df1)
        df1=plotModels1(fig,axes,db_folder,rows,cols,df1)
        df1=plotModels2(fig,axes,db_folder,rows,cols,df1)

    plt.close(fig)

    print("stopping main")
    return

###############################################################################

if __name__=="__main__":
    main()

###############################################################################
