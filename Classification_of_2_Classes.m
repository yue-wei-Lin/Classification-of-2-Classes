%  分辨兩種物體
cd F:\逢甲\智慧辨識檢測與應用\梁\train_test\my_train_test
trainData = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
testData = imageDatastore('test','IncludeSubfolders',true,'LabelSource','foldernames');
layers = [
    imageInputLayer([500 600 3])
    convolution2dLayer([5 5],30)
    reluLayer
    maxPooling2dLayer(3,'Stride',3)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

layers = [
    imageInputLayer([500 600 3])
    convolution2dLayer([5 5],30)
    reluLayer
    maxPooling2dLayer(3,'Stride',3)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
opts = trainingOptions('adam','InitialLearnRate',0.001,'MaxEpochs',50,'MiniBatchSize',16,'Plots','training-progress');
myNet = trainNetwork(trainData,layers,opts);
desiredLabel = testData.Labels;
predictedLabel = classify(myNet, testData);
accumarray = mean(desiredLabel == predictedLabel)
