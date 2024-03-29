function [acc_per_image,acc_per_class] = HDA_CDLPP_SP(trainS_features,trainS_labels,trainT_features,trainT_labels,test_features,test_labels,options);
num_iter = options.num_iter;
num_class = length(unique(trainS_labels));
trainT_features_gt = trainT_features;
trainT_labels_gt = trainT_labels;
fprintf('d=%d\n',options.ReducedDim);
for iter = 1:num_iter
    P = CDLPP(trainS_features,trainT_features,trainS_labels,trainT_labels,options);
        Ps = P(1:size(trainS_features,2),:);
        Pt = P(size(trainS_features,2)+1:end,:);
        trainS_proj = trainS_features*Ps;
        trainT_proj = trainT_features*Pt;
        train_proj = [trainS_proj;trainT_proj];
        train_labels = [trainS_labels,trainT_labels];
        test_proj = test_features*Pt;
        %% centralization and l2 norm
        proj_mean = mean(train_proj);
        train_proj = train_proj - repmat(proj_mean,[size(train_proj,1) 1 ]);
        trainS_proj = trainS_proj - repmat(proj_mean,[size(trainS_proj,1) 1 ]);
        trainT_proj = trainT_proj - repmat(proj_mean,[size(trainT_proj,1) 1 ]);
        test_proj = test_proj - repmat(proj_mean,[size(test_proj,1) 1 ]);
        train_proj = L2Norm(train_proj);
        trainS_proj = L2Norm(trainS_proj);
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
    targetClusterMeans = vgg_kmeans(double(test_proj'), num_class, classMeans')';
    targetClusterMeans = L2Norm(targetClusterMeans);
    distClusterMeans = EuDist2(test_proj,targetClusterMeans);
    expMatrix = exp(-distClassMeans);
    expMatrix2 = exp(-distClusterMeans);
    probMatrix = expMatrix./repmat(sum(expMatrix,2),[1 num_class]);
    probMatrix2 = expMatrix2./repmat(sum(expMatrix2,2),[1 num_class]);
    probMatrix = max(probMatrix,probMatrix2);
        [prob,predLabels] = max(probMatrix');
    %% svm
    elseif strcmp(options.classifier, 'svm')
        probMatrix = svm_classify(train_proj,test_proj,train_labels',test_labels');
        [prob,predLabels] = max(probMatrix');
    else
        fprintf('Classifier not defined!\n');
        return
    end
    %%
%     if iter==1
%         p = 1;%max(prob);
%     end
%     p = p*0.95;
%     trustable = prob>p;
    p=1-iter/num_iter;
    [sortedProb,index] = sort(prob);
    sortedPredLabels = predLabels(index);
    trustable = zeros(1,length(prob));
%     trustable =  prob>=sortedProb(min(length(prob),floor(length(prob)*p)+1));
    for i = 1:num_class
        thisClassProb = sortedProb(sortedPredLabels==i);
        if length(thisClassProb)>0
            trustable = trustable+ (prob>thisClassProb(floor(length(thisClassProb)*p)+1)).*(predLabels==i);
        end
    end
    pseudoLabels = predLabels;
    pseudoLabels(~trustable) = -1;

    %% Add pseudo-labeled target samples to training set
    trainT_features = [trainT_features_gt;test_features(logical(trustable),:)];
    trainT_labels = [trainT_labels_gt, pseudoLabels(logical(trustable))];
    %% calculate ACC
    acc_per_image = sum(predLabels==test_labels)/length(test_labels);
    for i = 1:num_class
        acc_per_class(i) = sum((predLabels == test_labels).*(test_labels==i))/sum(test_labels==i);
    end
    fprintf('Iteration=%d/%d, Acc:%0.3f,Mean acc per class: %0.3f\n', iter, num_iter, acc_per_image, mean(acc_per_class));
    if sum(trustable)>=length(prob)
        break;
    end
end