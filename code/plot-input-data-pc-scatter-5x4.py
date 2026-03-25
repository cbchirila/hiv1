import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

enzymes=["IN","PR","RT"]
variables = ['SlogP', 'TPSA', 'NumHBD', 'NumHBA', 'NumRotatableBonds']
variablesNames = ['SlogP', 'TPSA', 'HBD', 'HBA', 'RBN']
variableTitles = ["A","B","C","D","E"]
splits = [1,2,3,4]

os.system("cls")
print("starting")

rows = len(variables)
cols = len(splits)

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

sns.set_theme(style='whitegrid',palette="bright",font_scale=0.50)

fig,axes = plt.subplots(rows, cols, figsize=(5.0, 7.2), dpi=600)
fig.tight_layout(pad=0)

for enzyme in enzymes:
    print(enzyme)

    # for each split
    for col in range(cols):
        split = splits[col]

        data_list = []

        for phase in ["train","validation"]:
            data = pd.read_csv("./train-ch/data-decomposed/"+enzyme+"-PC-x_work-"+str(split)+"-"+phase+".csv")
            data["Phase"] = phase.capitalize()
            data_list.append(data)

        for phase in ["test"]:
            data = pd.read_csv("./train-ch/data-decomposed/"+enzyme+"-PC-x_"+phase+".csv")
            data["Phase"] = "Test"
            data_list.append(data)

        data_all = pd.concat(data_list)

        marker_size_map = {"Train":1, "Validation":1, "Test":1}
        #marker_symbol_map = {"Train":"x", "Validation":"x", "Test":"^"}

        # for each variable
        for row in range(rows):
            ax = axes[row,col]

            sns.scatterplot(
                ax=ax,
                data=data_all,
                x='ExactMW',
                y=variables[row],
                hue="Phase",
                hue_order=["Train", "Validation", "Test"],
                style="Phase",
                size=data_all["Phase"].map(marker_size_map),
                #sizes=[1,2,3],
                #marker=data_all["Phase"].map(marker_symbol_map),
                #markers=["o","x","s"],
                palette={"Train":"blue", "Validation":"orange", "Test":"lightgreen"},
                #palette=["blue","orange","lightgreen"],
                alpha=0.5,
                legend=False
            )

            ax.set_xlabel(variablesNames[row])
            ax.set_ylabel("Molecular Weight")

            # Afișare legendă doar pentru ultimul grafic
            #if split-1 == rows-1 and col_idx == cols - 1:
                # Obține manipulatorii și etichetele legendelor
                #handles, labels = ax.get_legend_handles_labels()  
                #print(handles)
                #print(labels)
                # Crează un dicționar pentru a elimina duplicatele
                #by_label = dict(zip(labels, handles))
                # Adaugă legenda pe grafic
                # ax.legend(
                #     by_label.values(),
                #     by_label.keys(),
                #     title='Legend',
                #     loc='upper left',
                #     legend=False
                # )  

            if row==0:
                ax.set_title(f"{split}")
                #ax.text(0.5,-0.5,f"{split}",fontsize=20,horizontalalignment="center",verticalalignment="center",transform=ax.transAxes)
            
            if col==0:
                ax.text(-0.70,0.5,variableTitles[row],horizontalalignment="center",verticalalignment="center",transform=ax.transAxes)

    plt.tight_layout()
    filename="./train-ch/data-plots/chembl-HIV1-PC-"+enzyme+"5x4"
    plt.savefig(filename+".png", bbox_inches="tight", orientation="portrait", dpi=600)
    #plt.savefig(filename+".eps", bbox_inches="tight", orientation="portrait", dpi=600)
    #plt.savefig(filename+".tif", bbox_inches="tight", orientation="portrait", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

    for i in range(rows):
        for j in range(cols):
            axes[i][j].cla()

fig.clf()
plt.close(fig)

print("stopping")
