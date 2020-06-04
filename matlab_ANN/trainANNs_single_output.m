clear
clc
close all


global ANNsetting
%global ANNtype
global ANNID
global blockMemory
global channels % siyu 7/3/2019
global DATA_DIR
global display_messages % siyu 7/3/2019
global fout
global highScale
global layers
global layerTypes
global lowScale
global memoryReduction
global percentCal
global percentVal
global predict_stations % siyu 7/3/2019
global SSE_desired % siyu 7/22/2019
global trainFunction 

% **********************************************************
% ********************* User Settings **********************
% **********************************************************

% 1. Define locations to be predicted:
% available:'Emmaton','Jersey Point','Collinsville', 'Rock Slough',
% 'Antioch', 'Mallard', 'LosVaqueros', 'Martinez', 'MiddleRiver', 'Vict
% Intake', 'CVP Intake', 'CCFB_OldR'
predict_stations = {'Emmaton','Jersey Point',...
                      'Collinsville', 'Rock Slough'};%,...
%                        'Antioch','Mallard','LosVaqueros',...
%                        'Martinez','MiddleRiver','Vict Intake',...
%                        'CVP Intake','CCFB_OldR'};
                   
available_inputs = {'SAC','Exp','SJR','DICU','Vern','SF_Tide','DXC'}; % do not change
% 2. Define variables to be used for prediction:
input_var = {'SAC','Exp','SJR','DICU','Vern','SF_Tide','DXC'};

% 3. Define directory to the input and output excel file:
% note: no blank space is allowed in DATA_DIR or FILE_NAME
% DATA_DIR = '/Users/siyuqi/Downloads';
DATA_DIR = '/Users/siyuqi/Documents/PhD/3_DSM2/Data_Code';
% DATA_DIR = 'D:/ANN/MATLAB/Data';
FILE_NAME = 'ANN_data.xlsx';

% 4. Define name of folder you want to save your ANN
ANNsetting ='single_output_ANN-0.1-0.9-8-2-1-80%-MEM-7-10-11'; % folder to put results in

% 5. (optional) Modify num of neurons and activation func in hidden layers
% Notes: current setting is [8, 2],
% this code only works for ANNs with 2 hidden layers
layers = {[8 2]};
layerTypes = {{'logsig','logsig','purelin'}};

% **********************************************************
% **************** User Settings Finished ******************
% **********************************************************


% ************** other settings *************************

fout=fopen('trainingSetup.out','w');
test_mode = false; % do not change
addpath('utils')

predict_stations=sort(predict_stations);

% ************** training setting *************************
display_messages = 1;
trainFunction = 'newff';
rand_seed = 1;
memoryReduction = 1;
%***************  ANN setup  *****************************
SSE_desired = 0;
lowScale =  0.1;
highScale = 0.9;   
blockMemory = 0;
percentCal = 0.8;
percentVal = 0.2;
learnFunction = 'trainbr';
weightLearnFunc = 'learngdm';
PerformanceFunc = 'msereg';

prefs = createModelPreferences(lowScale,highScale,blockMemory,percentCal,percentVal); % Siyu: deleted unnecessary preferences

fprintf(fout,'layers:%d %d %d\n',layers{1}(1),layers{1}(2),1);
fprintf(fout,'layerTypes:%s %s %s\n',char(layerTypes{1}{1}),char(layerTypes{1}{2}),char(layerTypes{1}{3}));
fprintf(fout,'trainFunction:%s\n',trainFunction);
fprintf(fout,'lowScale:%5.2f\n',lowScale);
fprintf(fout,'highScale:%5.2f\n',highScale);
fprintf(fout,'percentCal:%5.2f\n',percentCal);
fprintf(fout,'percentVal:%5.2f\n',percentVal);




%% load data
rng(rand_seed);

[input_ori, output_ori,predict_stations,input_text] = load_data(input_var,test_mode,fullfile(DATA_DIR,FILE_NAME),predict_stations);

channels=length(predict_stations);

% scale inputs and outputs
input0 = createModelInputs(input_ori,input_text,input_var,lowScale,highScale);
[outputs_info,outputs] = createModelOutputStructure(output_ori,prefs,predict_stations,display_messages);

width = size(input_ori,2);

fclose(fout);

if (display_messages) 
    fprintf('  Selecting random validation data with seed %d\n',rand_seed);
end

vset = rand(1,width);

if (display_messages)
    fprintf('  Creating validation set\n');
end    

%% modelSetup
pVal = percentCal + percentVal;
valset = input0(:,vset >= percentCal & vset < pVal);
inputs = input0(:,vset < percentCal);
outset={};
tgtset={};

for i = 1:channels
    assert(isequal(size(outputs{i}),size(vset)), ...
        ['Dimension mismatch between output data and'...
        'random selection matrix at channel %d, '...
        'output size is %s, but rand matrix size is %s'],...
        i,mat2str(size(vset)),mat2str(size(outputs{i}))) % siyu 07/23/2019

    assert(length(outputs{i})<=find(vset,1,'last'), ...
        ['Trying to select data No. %d from a %d-element matrix'...
        ' at channel %d'],...
        find(vset,1,'last'),length(outputs{i}),i) % siyu 07/23/2019

    
    tgtset{i} = struct('data',outputs{i}(vset >= percentCal & vset < pVal),...
        'a',outputs_info(i).a,...
        'b',outputs_info(i).b);
    outset{i} = struct('data',outputs{i}(vset < percentCal),...
        'a',outputs_info(i).a,...
        'b',outputs_info(i).b);
end
    
if (display_messages)
    fprintf('  Validation set created\n');
end

key_set = {'rock slough','rockslough','old river @ rock slough',...
            'emmaton',...
            'jersey point','jerseypoint',...
            'antioch',...
            'collinsville',...
            'mallard','mallard island',...
            'los vaqueros','losvaqueros',...
            'martinez',...
            'middle river','MiddleRiver',...
            'victoria cannal','Vict Intake',...
            'cvp intake',...
            'clfct forebay',...
            'clfct forebay intake',...
            'x2'};
value_set = {'ORRSL','ORRSL','ORRSL',...
            'EMM',...
            'JP','JP',...
            'antioch',...
            'CO',...
            'Mallard','Mallard',...
            'LosVaqueros','LosVaqueros',...
            'MTZ',...
            'MidR_intake','MidR_intake'...
            'Victoria_intake','Victoria_intake',...
            'CVP_intake',...
            'CCFB',...
            'CCFB_intake',...
            'X2'};
abbrev_dict = containers.Map(key_set,value_set);

for loc_id = 1:channels
    % valset: x_test; tgtset: y_test;
    VV.P = valset;
    VV.T = tgtset{loc_id}.data;
    VV.Pi = zeros(width,0);
    VV.Ai = zeros(width,0);
    try
        loc=abbrev_dict(lower(predict_stations{loc_id}));
    catch
        temp=predict_stations{loc_id};
        if length(temp)>=5
            loc=replace(temp(1:5),' ','');
        else
            loc=temp;
        end
    end
    trainANN(loc,inputs,outset{loc_id},VV);
end