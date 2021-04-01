function outmodel = trainANN(location,inputs,outputs,VV)

global display_messages
global ANNsetting
global percentCal
global layers
global layerTypes

if nargin ~= 4
    error('Usage must include location ID, inputs, outputs and VV');
end

%     siyu commented out two lines below
% msgLevel = 2;
% fout=fopen('trainingProcess.out','w'); 



if (display_messages)
    fprintf(['Creating network for: ',location,'\n']);
end


[net,trainResult, ANNconfig] = createModelNetwork(inputs,outputs,percentCal,display_messages,VV);%shengjun 12/8/2004

prefs.ANNconfig=ANNconfig; %11/18/2005

if (display_messages)
    fprintf(['Completed creating network for: ',location,'\n']);
end 

model = struct('name',      location,...
              'input',      inputs,...
              'output',     outputs,...
              'prefs',      prefs,...
              'network',    net,...
              'trainResult',struct('tr',trainResult{1},'Y',trainResult{2},'E',trainResult{3},...
                                   'Pf',{trainResult{4}},'Af',{trainResult{5}}));%shegnjun 2/17/2006
                  %'ANNID',     annid,... %siyu 7/3/2019
                  %'description',description,... % siyu 7/3/2019
                  %'inputRange', inputRange,... % siyu 7/3/2019


filename=['ANNmodel-', location, '.mat'];
filename= strrep(filename,' ','');
createNewF90Network(model); %shengjun
save(filename, 'model','location'); % siyu del 'type' 7/3/2019
disp(['Saved ',filename]);  

% siyu modified below 7/3/2019
% 
% tempID=strrep(ANNID,' ','');

if ~exist(['./network/',  ANNsetting,'/',location], 'dir')
   mkdir(['./network/',  ANNsetting,'/',location]);
end

movefile('*.mat',['./network/',  ANNsetting,'/',location],'f')
if length(layers{1})==3 && isequal(layerTypes , {{'logsig','logsig','purelin'}})
    movefile('*.f90',['./network/',  ANNsetting,'/',location],'f')
end
fclose('all');
dirStatus=size(dir('*.out'));
if(dirStatus(1)>0)      
%     movefile('*.out',['./network/',  ANNsetting],'f') \
%     mkdir(['./network/',  ANNsetting,'/',tempID,'/',location]);
    movefile('*.out',['./network/',  ANNsetting,'/'],'f')
end
  
% pack % siyu commented out 7/3/2019