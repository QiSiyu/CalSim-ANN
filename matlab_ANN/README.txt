--------------Introduction--------------
The MATLAB® codes reads data from excel files, normalize data and train ANNs.
There are two different ANN architectures:

1. Train one ANN on each of the stations respectively.
Each ANN contains two hidden layers with 8 and 2 neurons, followed by an output layer with 1 neuron.
Results will be written into the folder: network/$ANNsetting$/$station$, where $ANNsetting$ refers to the variable "ANNsetting" in "trainAll.m" and $station$ is the abbreviation of the station, for example: network/0.1-0.9-8-2-1-80%-MEM-7-10-11/JP

2. Train one ANN on multiple selected outputs and this single ANN predicts more than 1 values at a time.
The ANN contains two hidden layers with 32 and 8 neurons (can be modified), followed by an output layer with 4 neurons.
Results will be written into the folder: network/$ANNsetting$/$station$, where $ANNsetting$ refers to the variable "ANNsetting" in "trainAll.m" and $station$ is the concatenated name of stations, for example: network/4_output_ANN-0.1-0.9-8-2-1-80%-MEM-7-10-11/CO_EMM_JP_ORRSL


------------Required packages------------
MATLAB ® neural network toolbox


---------------Data format---------------
1. Data must be in excel format;
2. Each row should be the time series/outputs for one day;
3. Put input data in the first sheet and output data in the second;
4. Date/time can either be included as the first column or be not included, but in both cases, it will be ignored.
5. Days with empty entries will be deleted automatically.

------------------Usage------------------
Run with default settings:
  1. open "trainANNs_single_output.m" for training multiple separate ANNs, or "trainANN_multi_output.m" for training a single integrated ANN.
  2. change variables within "User settings" section in the script;
  3. run this script.

Run post-processing scripts:
  1. open "testANNs_single_output.m" for separate ANNs case, or "testANN_multi_output.m" for training a single integrated ANN.
  2. change variables within "User settings" section in the script;
  3. run this script.

