% Baesline methods for Heterogeneous Domain Adaptation
%% Loading Data:
% Features are extracted using resnet101 pretrained on ImageNet without
% fine-tuning
clear all
addpath('./utils/');
data_dir = '../ZSL_CrossDomain/Office10/';
domains = {'caltech','amazon','dslr','webcam'};
%domains = {'CALTECH','AMAZON','WEBCAM'};
num_training_per_class_source = 20;
num_training_per_class_target = 3;
pcaDim = 0;
lppDim = 32;

T = 0;
for randseed = 1:10
    semi = 0; % use semi-supervised learning/unlabelled target data or not?
    %load('amazon-decaf(PCA)-to-webcam-surf.mat');
    for target_domain_index = 1:length(domains)
        %if target_domain_index == source_domain_index
        %    continue;
        %end
        fprintf('Target domain: %s\n',domains{target_domain_index});
        load([data_dir 'decaf/' domains{target_domain_index} '_decaf.mat']);
        %load([data_dir 'surf/' domains{target_domain_index} '_SURF_L10']);
        %feas = data';
        %feas = fts;
        %domainT_features = L2Norm(feas);
        domainT_features = feas;
        domainT_labels = labels';
        
        if pcaDim >0
            options.ReducedDim = pcaDim;
            [P_pca,~] = PCA(domainT_features,options);
            domainT_features = domainT_features*P_pca;
        end
        %domainT_features = L2Norm(domainT_features);
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
        trainT_features = domainT_features(logical(selector),:);
        trainT_labels = domainT_labels(logical(selector));
        test_features = domainT_features(logical(1-selector),:);
        test_labels = domainT_labels(logical(1-selector));
        %trainT_features = T;
        %trainT_labels = T_Label';
        %test_features = Ttest;
        %test_labels = Ttest_Label';
        num_class = length(unique(domainT_labels));
        %% Baseline method: using SVMt, only labeled target samples for training
        fprintf('Baseline method using SVM:\n');
        classifierType='svm';
        [acc_per_image_svm{target_domain_index}(randseed), acc_per_class_svm{target_domain_index}(randseed,:)]= func_recognition(trainT_features,test_features,trainT_labels,test_labels,classifierType);
        %% Baseline method: using 1NNt, only labeled target samples for training
        fprintf('Baseline method using 1NN:\n');
        classifierType='1nn';
        [acc_per_image_1nn{target_domain_index}(randseed), acc_per_class_1nn{target_domain_index}(randseed,:)]= func_recognition(trainT_features,test_features,trainT_labels,test_labels,classifierType);
        %% Baseline method: using NCt, only labeled target samples for training
        fprintf('Baseline method using NC:\n');
        classifierType='nc';
        [acc_per_image_nc{target_domain_index}(randseed), acc_per_class_1nn{target_domain_index}(randseed,:)]= func_recognition(trainT_features,test_features,trainT_labels,test_labels,classifierType);
    end
end
save(['./results_supervised_HDA/office10-surf2decaf-noL2norm-SVMt-semi-' num2str(semi) '-randseed-' num2str(randseed) '-PcaDim-' num2str(pcaDim) '-LppDim-' num2str(lppDim) '-T-' num2str(T) '.mat'],'acc_per_class*','acc_per_image*');