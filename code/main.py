import datetime
import logging
import socket
import sys

def config_log(model):
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
	logging.info("machine "+socket.gethostname().lower())
	return

def main():
    config_log("abc")
    import model_abc as model_abc

    config_log("cnn")
    import model_cnn as model_cnn

    config_log("dtc")
    import model_dtc as model_dtc

    config_log("gnb")
    import model_gnb as model_gnb

    config_log("gpc")
    import model_gpc as model_gpc

    config_log("knn")
    import model_knn as model_knn

    config_log("mlp")
    import model_mlp as model_mlp

    config_log("qda")
    import model_qda as model_qda

    config_log("rfc")
    import model_rfc as model_rfc

    config_log("svc")
    import model_svc as model_svc

    return

main()

