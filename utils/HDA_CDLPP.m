function [acc_per_image,acc_per_class] = HDA_CDLPP(trainS_features,trainS_labels,trainT_features,trainT_labels,test_features,test_labels,options);
num_iter = options.num_iter;
num_class = length(unique(trainS_labels));
trainT_features_gt = trainT_features;
trainT_labels_gt = trainT_labels;
trainS_features_gt = trainS_features;
trainS_labels_gt = trainS_labels;
fprintf('d=%d\n',options.ReducedDim);
for iter = 1:num_iter
    P = CDLPP(trainS_features,trainT_features,trainS_labels,trainT_labels,options);
    Ps = P(1:size(trainS_features,2),:);
    Pt = P(size(trainS_features,2)+1:end,:);
    trainS_proj = trainS_features*Ps;
    trainS_proj_gt = trainS_features_gt*Ps;
    trainT_proj = trainT_features*Pt;
    train_proj = [trainS_proj;trainT_proj];
    train_labels = [trainS_labels,trainT_labels];
    test_proj = test_features*Pt;
    %% centralization and l2 norm
    proj_mean = mean(train_proj);
    train_proj = train_proj - repmat(proj_mean,[size(train_proj,1) 1 ]);
    trainS_proj = trainS_proj - repmat(proj_mean,[size(trainS_proj,1) 1 ]);
    trainS_proj_gt = trainS_proj_gt - repmat(proj_mean,[size(trainS_proj_gt,1) 1]);
    trainT_proj = trainT_proj - repmat(proj_mean,[size(trainT_proj,1) 1 ]);
    test_proj = test_proj - repmat(proj_mean,[size(test_proj,1) 1 ]);
    train_proj = L2Norm(train_proj);
    trainS_proj = L2Norm(trainS_proj);
    trainS_proj_gt = L2Norm(trainS_proj_gt);
    trainT_proj = L2Norm(trainT_proj);
    test_proj = L2Norm(test_proj);
    %classifierType='1nn';
    %pred = func_recognition(domainS_proj,domainT_proj, domainS_labels,domainT_labels,classifierType);
    %% 1NN
    if strcmp(options.classifier, '1nn')
        distances = EuDist2(test_proj,train_proj);
        [minDist,ind] = min(distances');
        predLabels = train_labels(ind);
        expMatrix = exp(-minDist);
        prob = expMatrix;
        %% class means of distances
    elseif strcmp(options.classifier, 'mcd')
        distances = EuDist2(test_proj,train_proj);
        classMeanDist = zeros(size(distances,1),num_class);
        for i = 1:num_class
            classMeanDist(:,i) = mean(distances(:,train_labels==i),2);
        end
        expMatrix = exp(-classMeanDist);
        probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
        [prob,predLabels] = max(probMatrix');
        
        %% distance to class means
    elseif strcmp(options.classifier, 'nc')
        classMeans = zeros(num_class,options.ReducedDim);
        for i = 1:num_class
            classMeans(i,:) = mean(train_proj(train_labels==i,:));
        end
        classMeans = L2Norm(classMeans);
        distClassMeans = EuDist2(test_proj,classMeans);
        distClassMeansSource = EuDist2(trainS_proj_gt,classMeans);
        expMatrix = exp(-distClassMeans);
        probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
        [prob,predLabels] = max(probMatrix');
        expMatrixSource = exp(-distClassMeansSource);
        probMatrixSource = expMatrixSource./repmat(sum(expMatrixSource,2),[1 num_class]);
        [probSource,predLabelsSource] = max(probMatrixSource');
        %% svm
    elseif strcmp(options.classifier, 'svm')
        probMatrix = svm_classify(train_proj,test_proj,train_labels',test_labels');
        [prob,predLabels] = max(probMatrix');
    else
        fprintf('Classifier not defined!\n');
        return
    end
    %%
    if options.deltaT == 0
        p=1-iter/num_iter; % select progressively if options.deltaT == 0
        p = max(0,p);
    elseif options.deltaT <= 1
        p = 1-options.deltaT; % select a fixed fraction of target samples
    end
    [sortedProb,index] = sort(prob);
    sortedPredLabels = predLabels(index);
    trustable = zeros(1,length(prob));
    %trustable = prob>=sortedProb(min(length(prob),floor(length(prob)*p)+1));
    for i = 1:num_class
        thisClassProb = sortedProb(sortedPredLabels==i);
        nClass = length(thisClassProb);
        if nClass>0
            trustable = trustable+ (prob>=thisClassProb(min(nClass,floor(nClass*p)+1))).*(predLabels==i);
        end
    end
    pseudoLabels = predLabels;
    pseudoLabels(~trustable) = -1;
    %% Select source samples for projection learning in the next iteration
    p = 1-options.deltaS;
    [sortedProb,index] = sort(probSource);
    sortedPredLabels = predLabelsSource(index);
    trustableSource = zeros(1,length(probSource));
    for i = 1:num_class
        thisClassProb = sortedProb(sortedPredLabels==i);
        nClass = length(thisClassProb);
        if nClass>0
            trustableSource = trustableSource+ (probSource>=thisClassProb(min(nClass,floor(nClass*p)+1))).*(predLabelsSource==i);
        end
    end
    trainS_features = trainS_features_gt(logical(trustableSource),:);
    trainS_labels = trainS_labels_gt(logical(trustableSource));
    %% Add pseudo-labeled target samples to training set
    trainT_features = [trainT_features_gt;test_features(logical(trustable),:)];
    trainT_labels = [trainT_labels_gt, pseudoLabels(logical(trustable))];
    %% calculate ACC
    acc_per_image(iter) = sum(predLabels==test_labels)/length(test_labels);
    for i = 1:num_class
        acc_per_class(iter,i) = sum((predLabels == test_labels).*(test_labels==i))/sum(test_labels==i);
    end
    fprintf('Iteration=%d/%d, Acc:%0.3f,Mean acc per class: %0.3f\n', iter, num_iter, acc_per_image(iter), mean(acc_per_class(iter,:)));
end
