
# versiunea buna ce va fi folosita in lucrare

import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

enzymes=["IN","PR","RT"]

os.system("cls")
print("starting")

plt.rc('text',usetex=True)
plt.rc('font',family='serif')
plt.title('Heatmap')

sns.set_theme(style='whitegrid',palette="bright",font_scale=0.5)

rows=3
cols=1
fig,axes=plt.subplots(nrows=rows,ncols=cols,figsize=(3.6,7.2),dpi=600)
fig.tight_layout(pad=11)

for i in range(rows):
    enzyme=enzymes[i]
    print(enzyme)

    # Încarcă datele despre descriptori într-un DataFrame
    data = pd.read_csv("./train-ch/data-selected/chembl-HIV1-"+enzyme+"-PC-heatmap.csv")

    # Calculează matricea de corelație între descriptorii moleculari
    correlation_matrix = data.corr()

    # Generează harta de căldură (heatmap) fără valori
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=axes[i])

# Salvează graficul
filename="./train-ch/data-plots/chembl-HIV1-PC-heatmap-3x1"
plt.savefig(filename+".png", bbox_inches="tight", orientation="portrait", dpi=600)
#plt.savefig(filename+".eps", bbox_inches="tight", orientation="portrait", dpi=600)
#plt.savefig(filename+".tif", bbox_inches="tight", orientation="portrait", dpi=600, format="tiff", pil_kwargs={"compression":"tiff_lzw"})

for i in range(rows):
    axes[i].cla()

fig.clf()
plt.close(fig)

print("stopping")
