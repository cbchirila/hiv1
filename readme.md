
# Replication Package for paper "Harnessing Machine Learning Models to Repurpose Drugs Targeting HIV-1 Integrase, Protease, and Reverse Transcriptase"

This package reproduces the experiments described in the paper:
Harnessing Machine Learning Models to Repurpose Drugs Targeting HIV-1 Integrase, Protease, and Reverse Transcriptase

Ciprian-Bogdan Chirila, University Politehnica Timisoara, 2 V. Parvan Avenue, Timisoara 300223, Romania, chirila@cs.upt.ro

Luminita Crisan, "Coriolan Dragulescu" Institute of Chemistry, 24 M. Viteazu Avenue,Timisoara  300223, Romania, lumi_crisan@acad-icht.tm.edu.ro

Keywords: antiviral, drug repurposing, machine learning

# Requirements

For CPU processing
- Python 3.10.10
- TensorFlow 2.10.0
- Tested on Windows 10, Windows 11

For GPU processing:
- Python 3.10.10
- Nvidia Cuda 11.8.0
- cudnn 8.6.0.163
- tensorflow-2.10.0-cp310-cp310-win_amd64.whl
- Tested on Windows 10, Windows 11

# Installation

In the root folder of the uncompressed archive you have to run the following commands:

# For CPU processing
pip install tensorflow-cpu==2.10.0
pip install -r requirements.txt

# For GPU processing:
pip install tensorflow-2.10.0-cp310-cp310-win_amd64.whl
pip install -r requirements.txt

# Running the Experiment Step 1: Training the Base Models

In the root folder of the uncompressed archive you have to run the following commands:

run-l1.bat
- Trains the 10 base models and generates first level predictions.
- The training is made on 9 use case files coresponding to the 3 enzymes x 3 descriptors.
- The models train on the data from ChEMBL database located in ./train-ch/data/ folder. 
- The trained models predict the inputs from DrugCentral database located in ./pred-dc/data/ folder.

python ./code/models/metric-performance.py
- Computes the performance metrics.

python ./code/plot-performance-metrics.py 
- Compiles the tables with the performance metrics data.

python ./code/plot-met-dist-histplot.py 
- Plots the performance metrics distribution histogram.

# Output Step 1

- The predictions from ChEMBL database are located in ./train-ch/predictions/ folder.
- The predictions from DrugCentral database are located in ./pred-dc/predictions/ folder.
- The performance metrics from ChEMBL are located in ./train-ch/metrics/ folder.
- The performance metrics plots from ChEMBL are located in ./train-ch/metrics/metrics-plots/ folder.

# Running the Experiment Step 2 Training the Meta-Learner

run-l2.bat
- Trains the metalearner on 9578 base level model predictions with varying hyperparameters.
- The models train on the input data processed from the ChEMBL predictions located in ./train-ch-ml/data/ folder. 
- The trained models predict the input data processed from DrugCentral predictions located in ./pred-dc-ml/data/ folder.

# Output Step 2

- The predictions of ChEMBL database are located in ./train-ch-ml/predictions/ folder.
- The predictions of DrugCentral database are located in ./pred-dc-ml/predictions/ folder.

# Notes

If a GPU is available, TensorFlow 2.10 will use it automatically.
If not, it will fall back to CPU execution.
The code was tested using both TensorFlow CPU and GPU builds.
The run-all.bat script runs both steps in a single command.
