function [input_ori] = createModelInputs(input_ori,input_text,input_var,lowScale,highScale)
global fout

base_input_indices = zeros(length(input_var),1);
for i = 1:length(input_var)
    base_input_index = find(contains(input_text, input_var{i}), 1, 'first');
    if ~isempty(base_input_index)
        base_input_indices(i) = base_input_index;
    end
end
input_var(base_input_indices==0)=[];
base_input_indices = nonzeros(base_input_indices);


input_min = min(input_ori,[],2);
input_max = max(input_ori,[],2);

[input_ori,a,b] = createScaledData(input_ori,lowScale,highScale);

fprintf(fout,'\n\nInput Data Range:\n');

for i = 1:length(base_input_indices)
    fprintf(fout,'\n%25s: ',input_var{i});
    fprintf(fout,'min=%11.2f, max=%11.2f\n',...
                input_min(base_input_indices(i)),...
                input_max(base_input_indices(i)));
end

fprintf(fout,'\n\nInput Scaling factors:\n');

for i = 1:length(base_input_indices)
    fprintf(fout,'\n%25s: ',input_var{i});
    fprintf(fout,'a=%10.8f; b= %11.8f;\n',a(base_input_indices(i)),...
                b(base_input_indices(i)));
end