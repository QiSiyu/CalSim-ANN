% clear
% clc
% close all

global ANNsetting
%global ANNtype
global ANNID
global blockMemory
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
global output_stations % siyu 7/3/2019
global SSE_desired % siyu 7/22/2019
global trainFunction 
% global useDefaultSigma % siyu 7/3/2019

test_mode = false; % do not change

% **********************************************************
% ********************* User Settings **********************
% **********************************************************


% 1. Select one or more output stations from:
% 'Emmaton','Jersey Point','Collinsville', 'Rock Slough',
% 'Antioch', 'Mallard', 'LosVaqueros', 'Martinez', 'MiddleRiver', 'Vict
% Intake', 'CVP Intake', 'CCFB_OldR'
output_stations = {'Emmaton','Jersey Point'};%,...
%                       'Collinsville', 'Rock Slough',...
%                        'Antioch','Mallard','LosVaqueros',...
%                        'Martinez','MiddleRiver','Vict Intake',...
%                        'CVP Intake','CCFB_OldR'};
                  
% 2. Select one or more input variables from:
% 'SAC','Exp','SJR','DICU','Vern','SF_Tide','DXC'
input_var = {'SAC','Exp','SJR','DICU','Vern','SF_Tide','DXC'};

% 3. Define directory to the input and output excel file:
DATA_DIR = '/Users/siyuqi/Downloads/CalSim-ANN-master';
FILE_NAME = 'ANN_data.xlsx';

% 4. Define name of folder you want to save your ANN
ANNsetting ='multi_output_ANN-0.1-0.9-8-2-1-80%-MEM-7-10-11'; % folder to put results in

% 5. (optional) Modify num of neurons and activation func in hidden layers
% Notes: current setting is [8 * num of stations, 2 * num of stations],
% this code only works for ANNs with 2 hidden layers
layers = {[8 2]*length(output_stations)};
layerTypes = {{'logsig','logsig','purelin'}};

% **********************************************************
% **************** User Settings Finished ******************
% **********************************************************


%************** other settings *************************

fout=fopen('trainingSetup.out','w');

addpath('utils')

output_stations=sort(output_stations);

%************** training setting *************************
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

fprintf(fout,'layers:%d %d %d\n',layers{1}(1),layers{1}(2),length(output_stations));
fprintf(fout,'layerTypes:%s %s %s\n',char(layerTypes{1}{1}),char(layerTypes{1}{2}),char(layerTypes{1}{3}));
fprintf(fout,'trainFunction:%s\n',trainFunction);
fprintf(fout,'lowScale:%5.2f\n',lowScale);
fprintf(fout,'highScale:%5.2f\n',highScale);
fprintf(fout,'percentCal:%5.2f\n',percentCal);
fprintf(fout,'percentVal:%5.2f\n',percentVal);



%% load data
rng(rand_seed);

[input_ori, output_ori,output_stations,input_text] = load_data(input_var,test_mode,fullfile(DATA_DIR,FILE_NAME),output_stations);


% scale inputs and outputs
input0 = createModelInputs(input_ori,input_text,input_var,lowScale,highScale);
[outputs_info,outputs] = createModelOutputStructure(output_ori,prefs,output_stations,display_messages, false);


width = size(input_ori,2);

fclose(fout);

if (display_messages) 
    fprintf('  Selecting random validation data with seed %d\n',rand_seed);
end

vset = rand(1,size(input0,2));

if (display_messages)
    fprintf('  Creating validation set\n');
end    

%% modelSetup
pVal = percentCal + percentVal;
valset = input0(:,vset >= percentCal & vset < pVal);
inputs = input0(:,vset < percentCal);
outset={};
tgtset={};

assert(isequal(size(outputs{1},2),size(vset,2)), ...
    ['Dimension mismatch between output data and'...
    'random selection matrix, '...
    'output length is %s, but rand matrix length is %s'],...
    mat2str(size(vset,2)),mat2str(size(outputs{1},2))) % siyu 07/23/2019

tgtset{1} = struct('data',outputs{1}(:,vset >= percentCal & vset < pVal),...
    'a',outputs_info(1).a,...
    'b',outputs_info(1).b);
outset{1} = struct('data',outputs{1}(:,vset < percentCal),...
    'a',outputs_info(1).a,...
    'b',outputs_info(1).b);

    
if (display_messages)
    fprintf('  Validation set created\n');
end

% valset: x_test; tgtset: y_test;
VV.P = valset;
VV.T = tgtset{1}.data;
VV.Pi = zeros(width,0);
VV.Ai = zeros(width,0);

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

for i =1:length(output_stations)
    try
        output_stations{i}=abbrev_dict(lower(output_stations{i}));
    catch
        temp=output_stations{i};
        if length(temp)>=5
            output_stations{i}=replace(temp(1:5),' ','');
        else
            output_stations{i}=temp;
        end
    end
end

loc = strrep(strrep(strjoin(output_stations,'_'),' ','_'),'__','_');

trainANN(loc,inputs,outset{1},VV);

