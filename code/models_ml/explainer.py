
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sys
from .config import create_directory

class Explainer():
	def __init__(self,data,model):
		logging.info("explainer.init")
		self.data=data
		self.model=model
		return

	def explain(self):
		logging.info("explainer.explain")
		rng = np.random.default_rng(7)
		background=self.data.x["train"][np.random.choice(self.data.x["train"].shape[0],100,replace=False)]	
		explainer=shap.Explainer(self.model,background)
		shap_values=explainer(self.data.x["test"])

		# print(self.data.df.columns.to_list()[1:])
		folder=self.data.root+"/explanations"+self.data.suf+"/"
		create_directory(folder)

		df1=pd.DataFrame(shap_values.values,columns=self.data.df.columns.to_list()[1:])
		df2=pd.DataFrame(shap_values.base_values)
		df3=pd.DataFrame(shap_values.data,columns=self.data.df.columns.to_list()[1:])
		df1.to_csv(folder+self.data.enzyme.lower()+"-"+self.data.descriptor.lower()+"-"+self.data.model_selection.lower()+"-"+str(self.data.split)+"-values.csv",index=False)
		df2.to_csv(folder+self.data.enzyme.lower()+"-"+self.data.descriptor.lower()+"-"+self.data.model_selection.lower()+"-"+str(self.data.split)+"-base_values.csv",index=False)
		df3.to_csv(folder+self.data.enzyme.lower()+"-"+self.data.descriptor.lower()+"-"+self.data.model_selection.lower()+"-"+str(self.data.split)+"-data.csv",index=False)

		# shap.plots.beeswarm(shap_values,show=False)
		# plt.xticks(ticks=plt.xticks()[0],labels=self.data.df.columns.to_list()[1:],rotation=45)
		# plt.savefig(folder+"/beeswarm.png",bbox_inches="tight", orientation="portrait", dpi=100)
		# plt.close()
		#shap.plots.bar(shap_values,show=False)
		#shap.plots.waterfall(shap_values[0],show=False)

		shap.summary_plot(shap_values,self.data.x["test"],feature_names=self.data.df.columns.to_list()[1:],rng=rng,show=False)
		plt.savefig(folder+self.data.enzyme.lower()+"-"+self.data.descriptor.lower()+"-"+self.data.model_selection.lower()+"-"+str(self.data.split)+"-summary.png",bbox_inches="tight",orientation="portrait",dpi=600)
		plt.close()
		return
