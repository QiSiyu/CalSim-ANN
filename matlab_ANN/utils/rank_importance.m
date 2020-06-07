close all
addpath('../Save_fig_Toolbox')
addpath('utils')

ANNsetting ='multi_output_ANN-0.1-0.9-8-2-1-80%-MEM-7-10-11'; % folder to put results in
% ANNsetting ='single_output_ANN-0.1-0.9-8-2-1-80%-MEM-7-10-11'; % folder to put results in
% DATA_DIR = '/Users/siyuqi/Downloads';
DATA_DIR = '/Users/siyuqi/Documents/PhD/3_DSM2/Data_Code';
FILE_NAME = "ANN_data.xlsx";
predict_stations = {'Emmaton','Jersey Point','Collinsville', 'Rock Slough'};
lowScale=0.1;
highScale=0.9;
predict_stations=sort(predict_stations);
test_mode=false;
% [input_ori, output_ori,predict_stations,input_text] = load_data(input_var,test_mode,fullfile(DATA_DIR,FILE_NAME),predict_stations);
% [input0,a,b]=createScaledData(input_ori,0.1,0.9);
input_var = var(input0,1,2)';

listing = dir('network/single_output_ANN-0.1-0.9-8-2-1-80%-MEM-7-10-11');
value_set = {'rock slough',...
    'emmaton',...
    'jersey point',...
    'antioch',...
    'collinsville',...
    'mallard',...
    'los vaqueros',...
    'martinez',...
    'middle river',...
    'Vict Intake',...
    'cvp intake',...
    'clfct forebay',...
    'clfct forebay intake',...
    'x2'};
key_set = {'ORRSL',...
    'EMM',...
    'JP',...
    'antioch',...
    'CO',...
    'Mallard',...
    'LosVaqueros',...
    'MTZ',...
    'MidR_intake',...
    'Vict',...
    'CVP_intake',...
    'CCFB_',...
    'CCFB_intake',...
    'X2'};
abbrev_dict = containers.Map(key_set,value_set);
ii=1;

if contains(ANNsetting,'multi')
    fname = 'CO_EMM_JP_ORRSL';
    mat_dir = fullfile('./network/',  ANNsetting,fname,['ANNmodel-',fname,'.mat']);
    loaded_ANN_workspace = load(mat_dir);
    l1 = loaded_ANN_workspace.model.network.IW{1};
    l2 = loaded_ANN_workspace.model.network.LW{2,1};
    l3 = loaded_ANN_workspace.model.network.LW{3,2};
    cw = (l3*l2*l1).*input_var;
    figure()
    for ii = 1:4
        plot(cw(ii,:))
        hold on
    end
    for ii = 1:17:119
        xline(ii);
    end
    yline(0);
    set(gca,'xtick',9:17:127,'xticklabel',replace(input_text(1:17:end),'_',' '),'TickLength',[0 0])
    title('Multi-output ANN Feature Importance Plot','Interpreter','none')
    ylabel('Connection Weight')
    legend(predict_stations,'Location','south')
    set(gcf, 'Position', [100, 100, 600,250]);
    export_fig(['/Users/siyuqi/Dropbox/calsim3_ANN_report/report/images/','4out_rank','.pdf'],'-transparent')
else
    ranking = zeros(length(listing),119);
    while ii <= length(listing)
        fname = listing(ii).name;
        if ~abbrev_dict.isKey(fname) || ~any(strcmpi(predict_stations,abbrev_dict(fname)))
            listing(ii) = [];
        else
            mat_dir = fullfile('./network/',  ANNsetting,fname,['ANNmodel-',fname,'.mat']);
            loaded_ANN_workspace = load(mat_dir);
            l1 = loaded_ANN_workspace.model.network.IW{1};
            l2 = loaded_ANN_workspace.model.network.LW{2,1};
            l3 = loaded_ANN_workspace.model.network.LW{3,2};
            ranking(ii,1:size(l1,2)) = (l3*l2*l1).*input_var;
            ii=ii+1;
        end
    end
    ranking(ii:end,:)=[];
    figure()
    for ii = 1:4
        plot(ranking(ii,:))
        hold on
    end
    for ii = 1:17:119
        xline(ii);
    end
    yline(0);
    set(gca,'xtick',9:17:127,'xticklabel',replace(input_text(1:17:end),'_',' '),'TickLength',[0 0])
    title('Single-output ANN Feature Importance Plot','Interpreter','none')
    ylabel('Connection Weight')
    legend(predict_stations,'Location','south')
    set(gcf, 'Position', [100, 100, 600,200]);
    export_fig(['/Users/siyuqi/Dropbox/calsim3_ANN_report/report/images/','1out_rank','.pdf'],'-transparent')
end
