% This script imports the trained ANNs and uses them to predict outputs
% 
% Models are loaded to the cell array "nets". If users want to use a 
% network for prediction, e.g. ANN for Emmaton, do as following:
%
% net = nets.net_emmaton;
% predictions = net(your_inputs);
% 

%% global declaration
global DATA_DIR
global output_stations
global ANNsetting

%% options for users
% **********************************************************
% ********************* User Settings **********************
% **********************************************************

% 1. Select one or more output stations from:
% 'Emmaton','Jersey Point','Collinsville', 'Rock Slough',
% 'Antioch', 'Mallard', 'LosVaqueros', 'Martinez', 'MiddleRiver', 'Vict
% Intake', 'CVP Intake', 'CCFB_OldR'
output_stations = {'Emmaton'};%,'Jersey Point',...
%                        'Collinsville', 'Rock Slough',...
%                        'Antioch','Mallard','LosVaqueros',...
%                        'Martinez','MiddleRiver','Vict Intake',...
%                        'CVP Intake','CCFB_OldR'};
                   
% 2. Select one or more input variables from:
% 'SAC','Exp','SJR','DICU','Vern','SF_Tide','DXC'
input_var = {'SAC','Exp','SJR','DICU','Vern','SF_Tide','DXC'};

% 3. Define directory to the input and output excel file
% Note: no blank space is allowed in DATA_DIR or FILE_NAME
DATA_DIR = '/Users/siyuqi/Downloads/CalSim-ANN-master';
FILE_NAME = 'ANN_data.xlsx';

% 4. Define ANNsetting (the folder where the model is saved). Must be same
% as training:
ANNsetting ='single_output_ANN-0.1-0.9-8-2-1-80%-MEM-7-10-11';

% 5. define whether to normalize outputs or not.
% Note: if set to true, output ec values are normalized between lowScale
% and highScale.
normalize_ec = true;

% 6. (optional) Define saving data precision (number of digits after the
% decimal point)
save_precision = 6;

% 7. (optional) set test_number to number of days users want to examine. If
% set to inf or negative, all the available inputs are sent to ANN.
test_size = inf;


% **********************************************************
% **************** User Settings Finished ******************
% **********************************************************

%% define normalization parameters and add path
lowScale = 0.1;
highScale = 0.9;
% 6/5 siyu added
test_mode = false;
addpath('utils')

output_stations=sort(output_stations);

%% load network and predict
[input_ori, output_ori,output_stations] = load_data(input_var,test_mode,fullfile(DATA_DIR,FILE_NAME),output_stations);

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


for i = 1:length(output_stations)
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

input_scaled = createScaledData(input_ori,lowScale,highScale);
[output_scaled,output_slope,output_bias] = createScaledData(output_ori,lowScale,highScale);

[output_ANN_predictions,nets] = predict(output_stations,input_scaled);

if ~normalize_ec
    output_ANN_predictions = ...
        denormalize_output(output_ANN_predictions,output_slope,output_bias);
    output_scaled = ...
        denormalize_output(output_scaled,output_slope,output_bias);
end

% write results to text files
if ~exist(fullfile('./network',  ANNsetting,'test_results'), 'dir')
   mkdir(fullfile('./network',  ANNsetting,'test_results'));
end

test_location = strrep(strrep(strjoin(output_stations,'_'),' ','_'),'__','_');

write2txt(output_ANN_predictions,output_stations,save_precision,...
    fullfile('./network/',  ANNsetting,'test_results',...
    ['single_out_',test_location,'_ANN_predictions.txt']));
write2txt(output_scaled,output_stations,save_precision,...
    fullfile('./network/',  ANNsetting,'test_results',...
    ['single_out_',test_location,'_target_outputs.txt']));

clear input_scaled
%% ----- Support functions -----


% load trained ANN and calculate predictions
function [output_ANN_predictions,nets] = predict(locations,input_scaled)
global ANNsetting
nets = struct;
output_ANN_predictions = zeros(length(locations),size(input_scaled,2));
for ii = 1:length(locations)
    location = locations{ii};
    mat_dir = fullfile('./network/',  ANNsetting,...
        location,['ANNmodel-',location,'.mat']);
    try
        loaded_ANN_workspace = load(mat_dir);
    catch 
        if exist('./network/','dir')
            if exist(fullfile('./network/',  ANNsetting),'dir')
                if exist(fullfile('./network/',  ANNsetting,location),'dir')
                    error('Cannot find .mat file, please check folder %s',...
                        fullfile('./network/',  ANNsetting,location))
                end
                error('Please train the model first before testing.')
            end
            error(['Can''t load model. '...
                'Please type ''ANNsetting'' in the command window, '...
                'check if there is a folder with the same name '...
                'in ''networks'' folder'])
        else
            error('Please train networks before testing.')
        end
    end
    eval(['net_',lower(location),' = loaded_ANN_workspace.model.network;']);
    eval(['output_ANN_predictions(ii,:) = net_',lower(location),'(input_scaled);']);
    eval(['nets.net_',lower(location),'=net_',lower(location),';']);
    disp(['Model evaluated for ',location])
end
end


function values = denormalize_output(values,slope,bias)
values = (values - bias) ./ slope;
end

function write2txt(values,output_stations,save_precision,write_directory)
if size(values,1)==length(output_stations)
    values = values';
end

rowNames = strsplit(num2str(1:size(values,1)));
loc = strrep(pad(output_stations,save_precision+2,'_'),' ','_');
eval(['values = sprintfc(''%.',num2str(save_precision),'f'',values);']);
% values = strsplit(num2str(values));
T = array2table(values,...
    'VariableNames',loc,'RowNames',rowNames);
writetable(T,write_directory,'WriteRowNames',true,...
    'WriteVariableNames',true,'Delimiter','\t');
end