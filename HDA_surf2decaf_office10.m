% unsupervised domain adaptation
% using LPP
%% Loading Data:
% Features are extracted using resnet101 pretrained on ImageNet without
% fine-tuning
clear all
addpath('./utils/');
data_dir = '../data/Office10/';
domains = {'caltech','amazon','dslr','webcam'};
num_training_per_class_source = 20;
num_training_per_class_target = 3;
pcaDim = 0;
lppDim = 10;
alpha = 10;
lambda = 0.5; % fixed value, used in CDLPP.m
deltaS = 1;% fixed
deltaT = 0;
using_sp = 0;
classifierType = 'nc';
T = 5;
for randseed = 1:10
semi = 1; % use semi-supervised learning/unlabelled target data or not?
for source_domain_index = 1:length(domains)
    load([data_dir 'surf/' domains{source_domain_index} '_SURF_L10']);
    feas = fts;
    %load([data_dir 'decaf/' domains{source_domain_index} '_decaf.mat']);
    domainS_features = L2Norm(feas);
    %domainS_features = feas;
    domainS_labels = labels';
    % dimension reduction for source domain data
    if pcaDim >0 
        options.ReducedDim = pcaDim;
        [P_pca,~] = PCA(domainS_features,options);
        domainS_features = domainS_features*P_pca;
    end
    
    %% training samples selection from source data
    rng(randseed);
    selector = zeros(1,length(domainS_labels));
    for iClass = 1:length(unique(domainS_labels))        
        numThisClass = sum(domainS_labels==iClass);
        if numThisClass <= num_training_per_class_source
            selector(domainS_labels==iClass) = 1;
        else
            randVector= rand(1, numThisClass);
            [sorted, sortIndex] = sort(randVector);        
            selector(domainS_labels==iClass) = randVector<= sorted(num_training_per_class_source);
        end        
    end
    trainS_features = domainS_features(logical(selector),:);
    trainS_labels = domainS_labels(logical(selector));
    %sourceDataSplits{randseed}{source_domain_index} = selector;
    for target_domain_index = 1:length(domains)
        fprintf('Randseed = %d\n',randseed);
        fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
        load([data_dir 'decaf/' domains{target_domain_index} '_decaf.mat']);
        %load([data_dir 'surf/' domains{target_domain_index} '_zscore_SURF_L10']);
        %feas = fts;
        domainT_features = L2Norm(feas);
        %domainT_features = feas;
        domainT_labels = labels';
        
        if pcaDim >0 
            options.ReducedDim = pcaDim;
            [P_pca,~] = PCA(domainT_featuress,options);
            domainT_features = domainT_features*P_pca;
        end        
        %% training and testing samples selection from target data
        rng(randseed);
        selector = zeros(1,length(domainT_labels));
        for iClass = 1:length(unique(domainT_labels))        
            numThisClass = sum(domainT_labels==iClass);
            if numThisClass <= num_training_per_class_target
                selector(domainT_labels==iClass) = 1;
            else
                randVector= rand(1, numThisClass);
                [sorted, sortIndex] = sort(randVector);        
                selector(domainT_labels==iClass) = randVector<= sorted(num_training_per_class_target);
            end        
        end
        %dataSplits{randseed}{target_domain_index} = selector;
        trainT_features = domainT_features(logical(selector),:);
        trainT_labels = domainT_labels(logical(selector));
        test_features = domainT_features(logical(1-selector),:);
        test_labels = domainT_labels(logical(1-selector));        
        num_class = length(unique(domainT_labels));
        %% HDA with CDLPP
        clear options;
        options.ReducedDim = lppDim;      
        options.classifier = classifierType;
        options.alpha = alpha; 
        options.lambda = lambda;
        options.deltaS = deltaS;
        options.deltaT = deltaT;   
        options.num_iter = T;
        [acc_per_image{source_domain_index}{target_domain_index}{randseed},acc_per_class{source_domain_index}{target_domain_index}{randseed}] = HDA_CDLPP(trainS_features,trainS_labels,trainT_features,trainT_labels,test_features,test_labels,options);
    end
end
end
save(['./results_semi_HDA/office10-nozscore-surf2decaf-' classifierType  '-sp-' num2str(using_sp) '-alpha-' num2str(alpha) '-lambda-' num2str(lambda) '-deltaS-' num2str(deltaS) '-deltaT-' num2str(deltaT)  '-randseed-' num2str(randseed) '-PcaDim-' num2str(pcaDim) '-LppDim-' num2str(lppDim) '-T-' num2str(T) '.mat'],'acc_per_class*','acc_per_image*');
%save('./office10-dataSplits.mat', 'dataSplits', 'sourceDataSplits');