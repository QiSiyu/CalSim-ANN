close all

addpath('../Save_fig_Toolbox')
addpath('utils')

DATA_DIR = '/Users/siyuqi/Downloads';
FILE_NAME = "ANN_data2.xlsx";
input_var = {'SAC','Exp','SJR','DICU','Vern','SF_Tide','DXC'};
predict_stations = {'Emmaton','Jersey Point',...
'Collinsville', 'Rock Slough',...
'Antioch','Mallard','LosVaqueros',...
'Martinez','MiddleRiver','Vict Intake',...
'CVP Intake','CCFB_OldR'};
lowScale=0.1;
highScale=0.9;
predict_stations=sort(predict_stations);
test_mode=false;
% [input_ori, output_ori,predict_stations,input_text] = load_data(input_var,test_mode,fullfile(DATA_DIR,FILE_NAME),predict_stations);
key_values={'SAC 0','Exports','DXC','SJR','DICU','Vern EC','SF_Tide'};
titles={'Northern Net Flows','Delta Exports',...
        'Delta Cross Channel Gate Operation','San Joaquin River Flows',...
        'Net Delta Consumptive Use', 'EC at Vernalis',...
        'Tidal Energy'};
title_map=containers.Map(key_values,titles);
t = datetime(1940,10,01) + caldays(0:27391);
% for ii = 2:2
%     figure()
%     plot(t,input_ori((ii-1)*18+1,:))
%     if ii == 7
%         ylim([-0.1,1.1])
%     end
%     title(title_map(input_text{(ii-1)*18+1}), 'Interpreter', 'none')
%     set(gcf, 'Position', [100, 100, 600,150]);
%     ax = gca;
%     ax.YRuler.Exponent = 0;
%     export_fig(['/Users/siyuqi/Dropbox/calsim3_ANN_report',...
%                 '/future_works_for_proposal/images/',...
%                 replace(input_text{(ii-1)*18+1},' ','_'),'.pdf'],'-transparent')
% end
for ii = 1:12
    figure()
    plot(t,output_ori(ii,:))
%     if ii == 7
%         ylim([-0.1,1.1])
%     end
    title(predict_stations{ii}, 'Interpreter', 'none')
    set(gcf, 'Position', [100, 100, 600,150]);
    ax = gca;
    ax.YRuler.Exponent = 0;
    export_fig(['/Users/siyuqi/Dropbox/calsim3_ANN_report',...
                '/future_works_for_proposal/images/',...
                replace(predict_stations{ii},' ','_'),'.pdf'],'-transparent')
end