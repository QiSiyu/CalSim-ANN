function [outputs_info,outputs] = createModelOutputStructure(output,prefs,predict_locations,messages,seperate_outputs)

global fout
global channels % siyu 7/3/2019

if (nargin < 5)
    seperate_outputs = true;
    if (nargin < 4)
        messages = 0;
    end
end



% load output data from files
if (messages)
    fprintf('Creating output structure\n');
end


% siyu modified loop 7/3/2019
ot=cell(1,channels*seperate_outputs+1*(~seperate_outputs));
if seperate_outputs
    for i = 1:size(ot,2)
        if (messages)
            fprintf(['Preparing output data for ',predict_locations{i},'... \n']);
        end
        ot{i} = output(i,:);
    end
else
    if (messages)
        fprintf(['Preparing output data for ','all locations','... \n']);
    end
    ot{1} = output;
end
% 
% disp(min(outdata{1}));
% disp(max(outdata{1}));

% group output data into locations (Antioch, Emmaton, JerseyPt, etc)
% ot = {};
% for i = 1:size(out,1)
%     ot{i} = [];
%     for (j = 1:size(out,2))
%         ot{i} = [ot{i};out{i,j}];
%     end
%     % log test
%     %ot{i} = log(ot{i});
% end

% clear out


% scale output data
%fout=fopen('trainingSetup.out','a')
fprintf(fout,'\n\nOutput Scaling factors:\n');
% fprintf(fout,'\n %s: \n',output.name);
% fprintf(fout,'\n %s: \n',output.file); % siyu commented out 7/3/2019

if (messages)
    fprintf('Scaling output data\n');
end

outputs_info = [];

if seperate_outputs
    for i = 1:length(ot)
        fprintf(fout,'\n %s: \n',predict_locations{i}); % siyu 7/3/2019
        fprintf(fout,'\n min=%11.8f, max=%11.8f\n',min(ot{i}), max(ot{i}));

        [ot{i},a,b] = createScaledData(ot{i},prefs.lowScale,prefs.highScale);

        fprintf(fout,'\n a=%10.8f; b= %11.8f;\n',a,b);

        outputs_info(i).a = a;
        outputs_info(i).b = b;
        disp('a: ')
        disp(single(a));
        disp('b: ')
        disp(single(b));
    end
else
    min_vals = min(ot{1},[],2);
    max_vals = max(ot{1},[],2);
    [ot{1},a,b] = createScaledData(ot{1},prefs.lowScale,prefs.highScale);
    for i = 1:size(ot{1},1)
        fprintf(fout,'\n %s: \n',predict_locations{i}); % siyu 7/3/2019
        fprintf(fout,'\n min=%11.8f, max=%11.8f\n',min_vals(i),max_vals(i));
        fprintf(fout,'\n a=%10.8f; b= %11.8f;\n\n',a(i),b(i));
    end
    outputs_info(1).a = a;
    outputs_info(1).b = b;
    disp('a: ')
    disp(single(a));
    disp('b: ')
    disp(single(b));
end
%fclose(fout);



clear a b

outputs = ot;
