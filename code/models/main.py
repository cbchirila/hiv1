import datetime
import logging
import socket
import sys

from .config import create_directory

def config_log(model):
	create_directory("./logs/")
	logging.root.handlers = []
	logging.basicConfig(
		level=logging.DEBUG,
		format="%(asctime)s [%(levelname)s] %(message)s",   
		handlers=[
			logging.FileHandler("./logs/hiv1"+
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

def main():
	config_log("abc")
	from . import model_abc

	config_log("cnn")
	from . import model_cnn

	config_log("dtc")
	from . import model_dtc

	config_log("gnb")
	from . import model_gnb

	config_log("gpc")
	from . import model_gpc

	config_log("knn")
	from . import model_knn

	config_log("mlp")
	from . import model_mlp

	config_log("qda")
	from . import model_qda

	config_log("rfc")
	from . import model_rfc

	config_log("svc")
	from . import model_svc

	config_log("abc-plt")
	from . import model_abc_plt

	config_log("dtc-plt")
	from . import model_dtc_plt

	config_log("gnb-plt")
	from . import model_gnb_plt

	config_log("gpc-plt")
	from . import model_gpc_plt

	config_log("knn-plt")
	from . import model_knn_plt

	config_log("qda-plt")
	from . import model_qda_plt

	config_log("rfc-plt")
	from . import model_rfc_plt

	config_log("svc-plt")
	from . import model_svc_plt

	return

###############################################################################

if __name__=="__main__":
	main()

###############################################################################
