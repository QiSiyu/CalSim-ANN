function [net, trainRecord, ANNconfig] = createModelNetwork(inputs,outputs,percentCal,msgLevel,VV)%shegnjun 12/8/2004
% This function creates a neural network and trains it on the inputs 
% using the outputs as targets
%   inputs is a double array of values to use as inputs
%   outputs is a double array of values to use as targets
%   percentCalibration is the amount of the data to be used as calibration
%       the rest being used for validation (0.75 is typical)
%  
% Properties of the network:
%
% global useDefaultSigma  % siyu 7/3/2019
global layers
global layerTypes
global trainFunction 
global SSE_desired
global inputDelay
global memoryReduction
global trainingMethod


if (nargin < 5)
    msgLevel = 0;
end

% %************** training setting *************************
if(isempty(trainingMethod)) 
    trainingMethod ='trainbr';
end %or trainscg    

PerformanceFunc='msereg';

weightLearnFunc = 'learngdm';                                      
% %********************************************************

params=[];

if (nargin < 4 || isempty(params))
    learnFunction = trainingMethod;	  
    
    if(strcmp(learnFunction, 'trainscg')==1)
        max_fail = 2;	% shengjun changed to 80
        epochs = 5000; %use 5000

    elseif(strcmp(learnFunction, 'trainbr')==1)        
        max_fail = 2;
        epochs = 1000;
    else
        max_fail = 2;
        epochs = 1000;        
    end
    
% siyu deleted these options 7/3/2019
%     if(useDefaultSigma==1)
%       %Matlab default setting:  
% 	  sigma = 5.0e-5;
% 	  lambda = 5.0e-7;
%     elseif(useDefaultSigma==0)
%       %existing setting
% 	  sigma = 1.0e-3;
%       lambda = 1.0e-2;
%     end

    min_grad = 1.0e-8;%1.0e-6

    params = createModelParameters(layers,layerTypes,trainFunction,learnFunction,epochs,max_fail,min_grad); % siyu 7/3/2019
end


if(strcmp(trainFunction,'newff'))
    % default choice
    net = fitnet(params.layers,params.learnFunction);% siyu 07/30/2019
%     eval(['net =',params.trainFunction,'(minmax(inputs),params.layers,params.layerTypes,params.learnFunction,weightLearnFunc,PerformanceFunc);']);%shengjun add 8/6/2004
elseif(strcmp(trainFunction,'newfftd'))
    params.learnFunction='trainrp';
    eval(['net =',params.trainFunction,'(minmax(inputs),',inputDelay,',params.layers,params.layerTypes,params.learnFunction,weightLearnFunc,PerformanceFunc);']);
elseif(strcmp(trainFunction,'newelm'))        
    params.learnFunction='traingdx';
    eval(['net =',params.trainFunction,'(minmax(inputs),params.layers,params.layerTypes,params.learnFunction,weightLearnFunc,PerformanceFunc);']);
end

net.trainParam.epochs   = params.epochs;
net.trainParam.max_fail = params.max_fail;

% net.trainParam.sigma    = params.sigma; % siyu 7/3/2019
% net.trainParam.lambda   = params.lambda; % siyu 7/3/2019
net.trainParam.min_grad = params.min_grad;

if(strcmp(learnFunction, 'trainbr'))  
    net.trainParam.show = 1;    
else
    net.trainParam.show = 5;
end
    

if (msgLevel)
    fprintf('  Training network\n');
end  

%[net] = train(net,inputs,outputs,zeros(width,0),zeros(width,0),VV);

if(strcmp(trainFunction,'newelm'))   
	inputs = con2seq(inputs);
  	outputs = con2seq(outputs);
    VV.P = con2seq(valset);
    VV.T = con2seq(tgtset);
else
%    class(inputs)
end

net = init(net);%2/17/2006

ANNconfig=struct('layers', layers,...
    'layerTypes',layerTypes,...
    'trainFunc',trainingMethod,...
    'weightLearnFunc',weightLearnFunc,...
    'PerformanceFunc',PerformanceFunc,...
    'maxfail',        max_fail,...
    'MaxEpoch',       epochs);
%     'sigma',sigma,... % siyu 7/3/2019
%     'lambda',lambda,... % siyu 7/3/2019


if(memoryReduction>1)
    net.trainParam.mem_reduc = memoryReduction;    
end

if(isempty(SSE_desired))
    SSE_desired=0.1;
end

if(percentCal==1.0)
    net.trainParam.goal = SSE_desired;
        
    [net,tr,Y,E,Pf,Af] = train(net,inputs,outputs.data);
else
   net.trainParam.goal = SSE_desired;
   [net,tr,Y,E,Pf,Af] = train(net,inputs,outputs.data);
end

%% siyu added for printing MSE and MAPE
train_out = net(inputs);
if any(any(isnan(train_out)))
    disp('NaN in predictions')
end
if any(any(isnan(outputs.data)))
    disp('NaN in target output')
    
end

perc_err_train = abs((train_out-outputs.data)./outputs.data);
disp("train mean percentage error:");
disp(mean(perc_err_train,2));
disp("train MSE:");
disp(mean((train_out-outputs.data).^2,2));

test_out = net(VV.P);
perc_err_test = abs((test_out-VV.T)./VV.T);
disp("test mean percentage error:");
disp(mean(perc_err_test,2));

disp("test MSE:");
disp(mean((test_out-VV.T).^2,2));

trainRecord={tr,Y,E,Pf,Af};

net.userdata=ANNconfig;

clear inputs outputs VV Y E Pf Af 