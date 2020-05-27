function params = createModelParameters(layers,layerTypes,trainFunction,learnFunction,epochs,max_fail,min_grad)

params = struct('layers'        ,layers,...
                'layerTypes'    ,layerTypes,...
                'trainFunction' ,trainFunction,...
                'learnFunction' ,learnFunction,...
                'epochs'        ,epochs,...
                'max_fail'      ,max_fail,...
                'min_grad'      ,min_grad);
%                 'sigma'         ,sigma,... % siyu 7/3/2019
%                 'lambda'        ,lambda,... % siyu 7/3/2019
