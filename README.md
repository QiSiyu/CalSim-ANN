# CalSim-ANN
Multi-variate Dense Neural Networks designed to predict salinity at Sacramento-San Joaquin Delta. Two architectures are provided: single-output ANNs and multi-output (integrated) ANNs.


## What's in this repo
*  Code to train and test ANNs in both MATLAB and Python.
*  Training and test set in "ANN_data.xlsx".

## Dataset format
To train the ANN with your own data, you have to prepare your own dataset that satisfies the following requirements.
1. Data must be in excel format;
2. Each row should be the time series/outputs for one day;
3. Put input data in the first sheet and output data in the second;
4. Date/time can either be included as the first column or be not included, but in both cases, it will be ignored.
5. Days with empty entries will be deleted automatically.


## How to run in MATLAB
### Requirements
* MATLAB 2019(a) or higher
* MATLAB Deep Learning Toolbox

### Introduction
The MATLAB scripts can read data from xlsx files, normalize data and train ANNs. Please refer to matlab_tutorial.pdf in this repository for more detailed instruction.

There are two different ANN architectures:

1. Train one ANN for each station respectively.
Each ANN contains two hidden layers with 8 and 2 neurons, followed by an output layer with 1 neuron.
Results will be written into the folder: network/$ANNsetting$/$station$, where $ANNsetting$ refers to the variable "ANNsetting" in "trainANNs_single_output.m" or "trainANN_multi_output.m", and $station$ is the abbreviation of the station, for example: network/0.1-0.9-8-2-1-80%-MEM-7-10-11/JP

2. Train one integrated ANN on multiple selected outputs and this single ANN predicts more than 1 values at a time.
The ANN contains two hidden layers with 32 and 8 neurons (can be modified), followed by an output layer with 4 neurons.
Results will be written into the folder: network/$ANNsetting$/$station$, where $ANNsetting$ refers to the variable "ANNsetting" in "trainANNs_single_output.m" or "trainANN_multi_output.m" and $station$ is the concatenated name of stations, for example: network/4_output_ANN-0.1-0.9-8-2-1-80%-MEM-7-10-11/CO_EMM_JP_ORRSL


### Usage
For detailed instruction, please refer to "matlab_manual.pdf".

To train ANNs on the given dataset, simply do the following:
  1. Open "trainANNs_single_output.m" (or "trainANN_multi_output.m" for an integrated ANN).
  2. change variables within "User settings" section in the script as needed;
  3. run the script.


To test the trained ANNs on the given dataset, simply do the following:
  1. After running training scripts, open "testANNs_single_output.m" (or "testANN_multi_output.m" for an integrated ANN).
  2. Change variables within "User settings" section in the script as needed;
  3. Run the script.

## How to run in Python
### Requirements
* Python 3.6
* Tensorflow==1.15
* matplotlib==3.2.1
* scipy==1.4.1
* pandas==1.0.3
* numpy==1.18

Note: these packages are already available on Google Colab.

### Usage
Python code is intended to run on Google Colab. Please refer to colab_tutorial.pdf in this repository for more detailed instruction.
