
import datetime
import logging
import os
import socket
import sys

from sklearn.preprocessing import StandardScaler,MinMaxScaler

models=["abc","cnn","dtc","gnb","gpc","knn","mlp","qda","rfc","svc"]
models_plt=["abc-plt","dtc-plt","gnb-plt","gpc-plt","knn-plt","qda-plt","rfc-plt","svc-plt"]
enzymes=["IN","PR","RT"]
descriptors=["PC","ECFP4","MIX"]
model_selections=["skf","sgkf"]
scalers=[(StandardScaler(),"std")]
phases=["train","validation","test"]
splits=[1,2,3,4]
splits_no=len(splits)

###############################################################################

dbs_root="."

###############################################################################

def config_log(model):
	create_directory("./logs")
	logging.root.handlers = []
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s [%(levelname)s] %(message)s",    
		handlers=[
			logging.FileHandler(dbs_root+"/logs/hiv1"+
				"-"+datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")+
				"-"+socket.gethostname().lower()+
				"-"+model+
				".txt"),
			logging.StreamHandler(sys.stdout),
		]
	)
	logging.getLogger("matplotlib").setLevel(logging.WARNING)
	logging.info("machine "+socket.gethostname().lower())
	return

###############################################################################

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            return None
        return directory_path

###############################################################################
