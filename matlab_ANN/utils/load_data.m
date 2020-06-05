function [input_ori, output_ori,predict_locations,input_text] = load_data(input_vars,test_mode,filename,predict_locations)

tic
fprintf('Reading from %s... \n',filename)
[input_ori,input_text,~] = xlsread(filename,1);
[output_ori,output_text,~] = xlsread(filename,2);
toc

% extract first row as header
input_text = input_text(1,:);
output_text = output_text(1,:);

%   delete date/time column 
if isempty(input_text{1})
    if size(input_text,2)==size(input_ori,2)
        input_ori = input_ori(:,2:end);
    end
    input_text = input_text(:,2:end);
elseif (size(input_text,2)==size(input_ori,2)-1)||...
        contains(input_text{1},'date','IgnoreCase',true)||...
        contains(input_text{1},'time','IgnoreCase',true)
    input_ori = input_ori(:,2:end);
end

if isempty(output_text{1})
    if size(output_text,2)==size(output_ori,2)
        output_ori = output_ori(:,2:end);
    end
    output_text = output_text(:,2:end);
elseif (size(output_text,2)==size(output_ori,2)-1)||...
        contains(output_text{1},'date','IgnoreCase',true)||...
        contains(output_text{1},'time','IgnoreCase',true)
    output_ori = output_ori(:,2:end)';
else
    output_ori = output_ori';
end

%make sure matrix is oriented properly
% dim1: time series
% dim2: days
if size(input_ori,1) > size(input_ori,2)
    input_ori = input_ori';
end

%make sure matrix is oriented properly
if size(output_ori,1) > size(output_ori,2)
    output_ori = output_ori';
end


% siyu added if loop below  07/23/2019
if length(output_ori)<length(input_ori)
    warning(['Imported less output data than input data, '...
    'discarding last %d set(s) of input data'],length(input_ori)-length(output_ori))
    input_ori = input_ori(:,1:length(output_ori));
elseif length(output_ori)>length(input_ori)
    warning(['Imported less input data than output data,'...
    'discarding last %d set(s) of output data'],length(output_ori)-length(input_ori))
    output_ori = output_ori(:,1:length(input_ori));
end   


%remove days with empty inputs
output_ori = output_ori(:,any(input_ori,1));
input_ori = input_ori(:,any(input_ori,1),:);

%remove columns (in excel) with empty features
input_text(isnan(input_ori(:,end)))=[];
input_ori(isnan(input_ori(:,end)),:)=[];

%remove rows with empty entries
output_ori=output_ori(:,~any(isnan(input_ori), 1));
input_ori=input_ori(:,~any(isnan(input_ori), 1));

% remove days with empty output entries
input_ori(:,any(isnan(output_ori)))=[];
output_ori(:,any(isnan(output_ori)))=[];

fprintf('Loaded input dataset size: %d x %d\n',size(input_ori));
fprintf('Loaded output dataset size: %d x %d\n',size(output_ori));

% extract predict stations
stations_to_predict = zeros(length(predict_locations),1);
for ii = 1:length(predict_locations)
    try
        stations_to_predict(ii) = find(strcmpi(output_text(1,:), predict_locations(ii)));
    catch
        warning('IMPORTANT: Station %s cannot be found in dataset',predict_locations{ii});
    end
end
fprintf('%d station(s) not found in dataset\n', sum(stations_to_predict==0))

predict_locations(stations_to_predict==0)=[];
stations_to_predict(stations_to_predict==0)=[];

for i = 1:length(predict_locations)
    formatSpec = 'Found station %s in column #%d \n';
    fprintf(formatSpec,predict_locations{i},stations_to_predict(i));
end

try
    output_ori = output_ori(stations_to_predict,:);
catch
    warning('station indices to predict exceeds 12 or below 1.')
    disp(stations_to_predict)
end


% extract input variables
input_vars_index = zeros(length(input_vars),1);
for ii = 1:length(input_text)
    input_vars_index(ii)=contains(input_text(ii),input_vars,'IgnoreCase',true);
end

% save('for_test.mat','input_vars_index','input_text','output_text','stations_to_predict')

input_ori(input_vars_index==0,:)=[];

fprintf('Selected input dataset size: %d x %d\n',size(input_ori));
fprintf('Selected output dataset size: %d x %d\n',size(output_ori));

