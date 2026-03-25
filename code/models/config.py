import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold,TimeSeriesSplit

descriptors=[("PC",(119,1),500),("ECFP4",(1024,1),2000),("MIX",(1143,1),2500)]
enzymes=["IN","PR","RT"]
scalers=[
    (StandardScaler(),"std"),
	#(MinMaxScaler(),"minmax"),
	]
n_splits=4

model_selections=[
	(StratifiedKFold(n_splits=n_splits+1,shuffle=False),StratifiedKFold(n_splits=n_splits,shuffle=False),"skf"),
	#(TimeSeriesSplit(n_splits=n_splits+1),TimeSeriesSplit(n_splits=n_splits),"tss"),
    (StratifiedGroupKFold(n_splits=n_splits+1,shuffle=False),StratifiedGroupKFold(n_splits=n_splits,shuffle=False),"sgkf"),
	]

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            return None
        return directory_path
