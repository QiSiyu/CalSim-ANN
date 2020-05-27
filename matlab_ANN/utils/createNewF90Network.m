function ans = createNewF90Network (model,dir,inp,prefs)

net = model.network;

if nargin < 2
    dir = '';
end
if nargin < 3
    inp = model.input;
end
if nargin < 4
    prefs = model.prefs;
end

%% scale and format inputs
if isstruct(inp)
    inp = createModelInputStructure(inp,prefs);
end
if size(inp,1) > size(inp,2)
    inp = rot90(inp);
end
PD = mat2cell(inp,size(inp,1));
Q = size(inp,2);
R = net.inputs{1}.size;

inputnodes = net.layers{1}.dimensions;
hlayer1nodes = net.layers{2}.dimensions;
hlayer2nodes = net.layers{3}.dimensions;

%% input values
PD1 = PD{1,1,1};

%% initial weights
IW1 = net.IW{1,1};

%% layer weights
LW2 = net.LW{2,1};
LW3 = net.LW{3,2};

%% biases
BZ1 = net.b{1}(:,ones(1,Q));
BZ2 = net.b{2}(:,ones(1,Q));
BZ3 = net.b{3}(:,ones(1,Q));

name = model.name;

filename = [dir,'fnet_',name,'.f90'];
fid = fopen(filename,'w');
if (fid == -1)
    fprintf('Unable to create file');
    return;
end

fprintf(fid,['module fnet_',name,'\n']);
if length(model.output(1).a)>1
    fprintf(fid,['\n! a = ',num2str(model.output(1).a'),'\n! b = ',num2str(model.output(1).b'),'\n\n']);
else
    fprintf(fid,['\n! a = ',num2str(model.output(1).a),'\n! b = ',num2str(model.output(1).b),'\n\n']);
end
% 
% fprintf(fid,'\n! a = \n');
% fprintf(fid, '%.4f\n',model.output.a);
% fprintf(fid,'\n! b = \n');
% fprintf(fid, '%.4f\n',model.output.b);
fprintf(fid,'\nintrinsic Reshape\n');

fprintf(fid,['real, dimension(',num2str(inputnodes),',',num2str(R),') :: input = &\n']);
fprintf(fid,['  Reshape((/']);
count = 0;

nIW1 = rot90(fliplr(IW1)); % this flipping is done to accound for how the Reshape function reconstructs the matrix %%%rot
%%nIW1 = rot90(IW1,-1); %%%rot

%%%for i = 1:inputnodes
for i = 1:R %%%rot
%%%    for j = 1:R
    for j = 1:inputnodes %%%rot
        if (~(i==1 && j==1))
            fprintf(fid,',');            
        end
%%%        fprintf(fid,'%0.4f',IW1(i,j));
%%%        fprintf(fid,'%0.4f',nIW1(i,j));  %% 5-input 39 line limitation
%            fprintf(fid,'%0.16f',nIW1(i,j)); %make 16 significant digit
            try
                fprintf(fid,'%0.12f',nIW1(i,j)); %make 12 significant digit
            catch
                continue
            end
%%%        fprintf(fid,'%0.2f',nIW1(i,j)); %%%rot
        count = count + 1;
        if (count == 12)  %% 5-input 39 line limitation
%%%        if (count == 19)
            fprintf(fid,' &\n            ');
            count = 0;
        end
%         if (mod(j,10) == 0 & ~(i==inputnodes & j==R))
%             fprintf(fid,' &\n            ');
%         end
    end
end
fprintf(fid,['/),(/',num2str(inputnodes),',',num2str(R),'/))\n']);

nLW2 = rot90(fliplr(LW2)); %%%rot
%%nLW2 = rot90(LW2,-1); %%%rot

fprintf(fid,['real, dimension(',num2str(hlayer1nodes),',',num2str(inputnodes),') :: hidden1 = &\n']);
fprintf(fid,['  Reshape((/']);
%%%for i = 1:hlayer1nodes
for i = 1:inputnodes %%%rot
%%%    for j = 1:inputnodes
    for j = 1:hlayer1nodes %%%rot
        if (~(i==1 & j==1))
            fprintf(fid,',');            
        end
%%%        fprintf(fid,num2str(LW2(i,j)));
        fprintf(fid,num2str(nLW2(i,j))); %%%rot
    end
    if (i ~= hlayer1nodes)
        fprintf(fid,' &\n            ');
    end
end
fprintf(fid,['/),(/',num2str(hlayer1nodes),',',num2str(inputnodes),'/))\n']);

nLW3 = rot90(fliplr(LW3)); %%%rot
%%nLW3 = rot90(LW3,-1); %%%rot

fprintf(fid,['real, dimension(',num2str(hlayer2nodes),',',num2str(hlayer1nodes),') :: hidden2 = &\n']);
fprintf(fid,['  Reshape((/']);
%%%for i = 1:hlayer2nodes
for i = 1:hlayer1nodes %%%rot
%%%    for j = 1:hlayer1nodes
    for j = 1:hlayer2nodes %%%rot
        if (~(i==1 & j==1))
            fprintf(fid,',');            
        end
%%%        fprintf(fid,num2str(LW3(i,j)));
        fprintf(fid,num2str(nLW3(i,j))); %%%rot
    end
end
fprintf(fid,['/),(/',num2str(hlayer2nodes),',',num2str(hlayer1nodes),'/))\n']);

fprintf(fid,['real, dimension(',num2str(inputnodes),') :: bias1 = &\n']);
fprintf(fid,['  (/']);
for i = 1:inputnodes
    if i ~= 1
        fprintf(fid,',');            
    end
    fprintf(fid,num2str(BZ1(i)));
end
fprintf(fid,['/)\n']);

fprintf(fid,['real, dimension(',num2str(hlayer1nodes),') :: bias2 = &\n']);
fprintf(fid,['  (/']);
for i = 1:hlayer1nodes
    if i ~= 1
        fprintf(fid,',');            
    end
    fprintf(fid,num2str(BZ2(i)));
end
fprintf(fid,['/)\n']);

fprintf(fid,['real, dimension(',num2str(hlayer2nodes),') :: bias3 = &\n']);
fprintf(fid,['  (/']);
for i = 1:hlayer2nodes
    if i ~= 1
        fprintf(fid,',');            
    end
    fprintf(fid,num2str(BZ3(i)));
end
fprintf(fid,['/)\n']);

fprintf(fid,['contains\n']);
fprintf(fid,['subroutine fnet_',name,'_initall()\n']);
fprintf(fid,['end subroutine fnet_',name,'_initall\n']);
fprintf(fid,['subroutine fnet_',name,'_engine(inarray, outarray, init)\n']);
fprintf(fid,['  intrinsic MatMul, Size\n']);
fprintf(fid,['  real, dimension(:), intent(in) :: inarray\n']);
fprintf(fid,['  real, dimension(:), intent(inout) :: outarray\n']);

ArraySize= size(model.input,1); % siyu 7/3/2019
fprintf(fid,['  real, dimension(',num2str(ArraySize),') :: inarray2\n']); % shengjun 8/3/2004

fprintf(fid,['  real (kind=8), dimension(',num2str(inputnodes),') :: layer1\n']);
fprintf(fid,['  real (kind=8), dimension(',num2str(hlayer1nodes),') :: layer2\n']);
fprintf(fid,['  real (kind=8), dimension(',num2str(hlayer2nodes),') :: layer3\n']);
fprintf(fid,['  integer , intent(inout) :: init\n']);
fprintf(fid,['  integer :: i, j\n']);
%fprintf(fid,['  do i = 1, 72\n']);  % will need to be changed if other than 72
fprintf(fid,['  do i = 1, ',num2str(ArraySize),'\n']); % shengjun 8/3/2004

fprintf(fid,['    inarray2(i) = inarray(',num2str(ArraySize+1),'-i)\n']);%shengjun 8/3/2004
fprintf(fid,['  end do\n']);
fprintf(fid,['  layer1 = MatMul(input,inarray2)\n']);
fprintf(fid,['  layer1 = layer1 + bias1\n']);
fprintf(fid,['  do i = 1, Size(layer1,1)\n']);
fprintf(fid,['    layer1(i) = 1.0 / (1.0 + DEXP(-1.0 * layer1(i)))\n']);
fprintf(fid,['  end do\n']);
fprintf(fid,['  layer2 = MatMul(hidden1,layer1)\n']);
fprintf(fid,['  layer2 = layer2 + bias2\n']);
fprintf(fid,['  do i = 1, Size(layer2,1)\n']);
fprintf(fid,['    layer2(i) = 1.0 / (1.0 + DEXP(-1.0 * layer2(i)))\n']);
fprintf(fid,['  end do\n']);
fprintf(fid,['  layer3 = MatMul(hidden2,layer2)\n']);
fprintf(fid,['  layer3 = layer3 + bias3\n']);
fprintf(fid,['  outarray(1) = layer3(1)\n']);
fprintf(fid,['end subroutine fnet_',name,'_engine\n']);
fprintf(fid,['end module fnet_',name,'\n']);

fclose(fid);
% 
% %% Input Layer
% 	AD1 = IW1 * PD1; % matrix multiply weights [8 x R] * [R x Q] --> [8 x Q]
% 	AD1 = AD1 + BZ1; % add biases
% 	a = 1 ./ (1+exp(-AD1)); % logsig
%     %% reassign NaNs to either -1 or +1
% 		k = find(~finite(a));
% 		a(k) = sign(AD1(k));
% 	AD1 = a;
% 
% %% Hidden Layer 1
% 	AD2 = LW2 * AD1; % matrix multiply weights [2 x 8] * [8 x Q] --> [2 x Q]
% 	AD2 = AD2 + BZ2; % add biases
% 	a = 1 ./ (1+exp(-AD2)); % logsig
%     %% reassign NaNs to either -1 or +1
% 		k = find(~finite(a));
% 		a(k) = sign(AD2(k));
% 	AD2 = a;
% 
% %% Hidden Layer 2
% 	AD3 = LW3 * AD2; % matrix multiply weights [1 x 2] * [2 x Q] --> [1 x Q]
% 	AD3 = AD3 + BZ3; % add biases
% ans = AD3;
% 
% %% unscale results
