clear all
% read dataset
imds = imageDatastore('dataset_10000\','IncludeSubfolders',true,'LabelSource','foldernames','FileExtensions',{'.mat'});

% split data to training and testing sets.
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

% read data into memory
imdsTrain.Labels = categorical(imdsTrain.Labels);
imdsTrain.ReadFcn = @readFcnMatFile;
 
imdsTest.Labels = categorical(imdsTest.Labels);
imdsTest.ReadFcn = @readFcnMatFile;

labelCount = countEachLabel(imds)

% training options configuration
batchSize   = 128;
ValFre      = fix(length(imdsTrain.Files)/batchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',batchSize, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch',...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',4,...
    'LearnRateDropFactor',0.1,...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',ValFre, ...
    'ValidationPatience',Inf, ...
    'Verbose',true ,...
    'VerboseFrequency',ValFre,...
    'Plots','training-progress',...
    'ExecutionEnvironment','multi-gpu');    % set up the GPU environment for training
% start the training process

% load the network 
RF_UAVNet
trainednet = trainNetwork(imdsTrain,lgraph,options);

% measure the accuracy on the testing set.
tic 
YPred = classify(trainednet,imdsTest,'MiniBatchSize',512,'ExecutionEnvironment','gpu');
toc
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest)

% store the output information
trainednetInfo = {};
trainednetInfo{1,1} = trainednet;
trainednetInfo{1,2} = YTest;
trainednetInfo{1,3} = YPred;
trainednetInfo{1,4} = accuracy;
trainednetInfo{1,5} = imdsTrain;
trainednetInfo{1,6} = imdsTest;
save('trainednet.mat','trainednetInfo')