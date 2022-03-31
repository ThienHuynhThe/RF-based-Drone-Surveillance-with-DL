% designing the deep convolutional neural network of multi-class operation
% mode classification.
lgraph = layerGraph();

nFilters    = 8;    % number of filters in each group
nGroups     = 8;    % number of groups defined in each layer
nClass      = 10;   % number of classes 

tempLayers = [
    imageInputLayer([1 10000 2],"Name","imageinput")
    convolution2dLayer([1 5],64,"Name","conv","Padding","same","Stride",[1 5])
    batchNormalizationLayer("Name","batchnorm_1")
    eluLayer(1,"Name","elu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([1 2],"Name","maxpool_1","Padding","same","Stride",[1 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 3],nFilters,nGroups,"Name","groupedconv_1","Padding","same","Stride",[1 2])
    %batchNormalizationLayer("Name","batchnorm_2")
    eluLayer(1,"Name","elu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalAveragePooling2dLayer("Name","gapool_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([1 2],"Name","maxpool_2","Padding","same","Stride",[1 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 3],nFilters,nGroups,"Name","groupedconv_2","Padding","same","Stride",[1 2])
    %batchNormalizationLayer("Name","batchnorm_3")
    eluLayer(1,"Name","elu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalAveragePooling2dLayer("Name","gapool_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([1 2],"Name","maxpool_3","Padding","same","Stride",[1 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 3],nFilters,nGroups,"Name","groupedconv_3","Padding","same","Stride",[1 2])
    %batchNormalizationLayer("Name","batchnorm_4")
    eluLayer(1,"Name","elu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([1 2],"Name","maxpool_4","Padding","same","Stride",[1 2]);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalAveragePooling2dLayer("Name","gapool_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    groupedConvolution2dLayer([1 3],nFilters,nGroups,"Name","groupedconv_4","Padding","same","Stride",[1 2])
    %batchNormalizationLayer("Name","batchnorm_5")
    eluLayer(1,"Name","elu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    globalAveragePooling2dLayer("Name","gapool_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = globalAveragePooling2dLayer("Name","gapool_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(5,"Name","depthcat")
    fullyConnectedLayer(nClass,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"elu_1","maxpool_1");
lgraph = connectLayers(lgraph,"elu_1","groupedconv_1");
lgraph = connectLayers(lgraph,"maxpool_1","addition_1/in2");
lgraph = connectLayers(lgraph,"elu_2","gapool_2");
lgraph = connectLayers(lgraph,"elu_2","addition_1/in1");
lgraph = connectLayers(lgraph,"gapool_2","depthcat/in5");
lgraph = connectLayers(lgraph,"addition_1","maxpool_2");
lgraph = connectLayers(lgraph,"addition_1","groupedconv_2");
lgraph = connectLayers(lgraph,"maxpool_2","addition_2/in2");
lgraph = connectLayers(lgraph,"elu_3","addition_2/in1");
lgraph = connectLayers(lgraph,"elu_3","gapool_5");
lgraph = connectLayers(lgraph,"addition_2","maxpool_3");
lgraph = connectLayers(lgraph,"addition_2","groupedconv_3");
lgraph = connectLayers(lgraph,"gapool_5","depthcat/in4");
lgraph = connectLayers(lgraph,"maxpool_3","addition_3/in2");
lgraph = connectLayers(lgraph,"elu_4","addition_3/in1");
lgraph = connectLayers(lgraph,"elu_4","gapool_4");
lgraph = connectLayers(lgraph,"addition_3","maxpool_4");
lgraph = connectLayers(lgraph,"addition_3","groupedconv_4");
lgraph = connectLayers(lgraph,"maxpool_4","addition_4/in2");
lgraph = connectLayers(lgraph,"gapool_4","depthcat/in3");
lgraph = connectLayers(lgraph,"elu_5","addition_4/in1");
lgraph = connectLayers(lgraph,"elu_5","gapool_3");
lgraph = connectLayers(lgraph,"gapool_1","depthcat/in1");
lgraph = connectLayers(lgraph,"gapool_3","depthcat/in2");

analyzeNetwork(lgraph)