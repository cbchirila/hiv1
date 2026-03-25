
import logging
import socket
from .translator_hp import translate_data_hp
from .translator_mean import translate_data_mean
from .mean_predictions import mean_predictions
from .variance_skf_sgkf import variance_skf_sgkf
from .model_lh import run_lh
from .config import models
from .config import models_plt
from .config import dbs_root

def main():

	# Translate train and prediction data for meta learner with mean predictions values
	translate_data_mean()

	# Train meta-learner
	run_lh(models=models+models_plt)
	mean_predictions(["chembl","drugcentral"],[dbs_root+"/train-ch-ml/predictions-18/",dbs_root+"/pred-dc-ml/predictions-18/"])

	run_lh(models=models)
	mean_predictions(["chembl","drugcentral"],[dbs_root+"/train-ch-ml/predictions-10/",dbs_root+"/pred-dc-ml/predictions-10/"])

	run_lh(models=models_plt)
	mean_predictions(["chembl","drugcentral"],[dbs_root+"/train-ch-ml/predictions-08/",dbs_root+"/pred-dc-ml/predictions-08/"])

	# Leave One Out (LOO)
	for model in models+models_plt:
		logging.info("Leave One Out %s",model.upper())
		run_lh(models=models+models_plt,removed_model=model)
		mean_predictions(["chembl","drugcentral"],[
			dbs_root+"/train-ch-ml/predictions-loo-"+model+"/",
			dbs_root+"/pred-dc-ml/predictions-loo-"+model+"/"])

	###########################################################################

	# Translate train and prediction data for meta learner with all hyper-parameters prediction values
	translate_data_hp()
	
	# Train meta-learner with shap enabled
	run_lh(models=models+models_plt,hp=True,shap=True)
	mean_predictions(["chembl","drugcentral"],[dbs_root+"/train-ch-ml/predictions-18-hp/",dbs_root+"/pred-dc-ml/predictions-18-hp/"])
	variance_skf_sgkf(["chembl"],[dbs_root+"/train-ch-ml/predictions-18-hp/"])

	run_lh(models=models,hp=True,shap=True)
	mean_predictions(["chembl","drugcentral"],[dbs_root+"/train-ch-ml/predictions-10-hp/",dbs_root+"/pred-dc-ml/predictions-10-hp/"])

	# Y scrambled (YS)
	run_lh(models=models+models_plt,hp=True,ys=True)
	mean_predictions(["chembl","drugcentral"],[dbs_root+"/train-ch-ml/predictions-18-hp-ys/",dbs_root+"/pred-dc-ml/predictions-18-hp-ys/"])

	return

if __name__=="__main__":
	main()
