% unsupervised domain adaptation
% using LPP
%% Loading Data:
% Features are extracted using resnet101 pretrained on ImageNet without
% fine-tuning
clear all
addpath('./utils/');
data_dir = './data/text/';
domains = {'EN','FR','GR','IT','SP'};
num_training_per_class_source = 100;
num_training_per_class_target = 10;
% num_test = 500;
num_test_per_class = 500;
pcaDim = 0;
lppDim = 10;
alpha = 10;
lambda = 0.5;
deltaS = 1;
deltaT = 0;
classifierType = 'nc';
T = 5;
using_sp = 0;
semi = 1; % use semi-supervised learning/unlabelled target data or not?
for randseed = 1:10
    for source_domain_index = 1:4
        %load([data_dir 'surf/' domains{source_domain_index} '_zscore_SURF_L10']);
        %feas = fts;
        load([data_dir domains{source_domain_index}]);
        domainS_features = L2Norm(feature_pca);
        %domainS_features = feas;
        domainS_labels = label';
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
        sourceDataSplits{randseed}{source_domain_index} = selector;
        for target_domain_index = 5
            fprintf('Randseed=%d\n',randseed);
            fprintf('Source domain: %s, Target domain: %s\n',domains{source_domain_index},domains{target_domain_index});
            %load([data_dir 'decaf/' domains{target_domain_index} '_decaf.mat']);
            load([data_dir domains{target_domain_index}]);
            feas = feature_pca;
            domainT_features = L2Norm(feas);
            %domainT_features = feas;
            domainT_labels = label';
            
            if pcaDim >0
                options.ReducedDim = pcaDim;
                [P_pca,~] = PCA(domainT_features,options);
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
                    selector_test(domainT_labels==iClass) = randVector > sorted(end-num_test_per_class);
                end
            end
            dataSplits{randseed}{target_domain_index} = selector;
            trainT_features = domainT_features(logical(selector),:);
            trainT_labels = domainT_labels(logical(selector));
            test_features = domainT_features(logical(selector_test),:);
            test_labels = domainT_labels(logical(selector_test));
            num_class = length(unique(domainT_labels));
            %% HDA with CDLPP
            clear options;
            options.ReducedDim = lppDim;
            options.classifier = classifierType;
            options.alpha = alpha;
            options.lambda = lambda;
            options.deltaS = deltaS;
            options.deltaT = deltaT;
            options.NeighborMode = 'KNN';
            options.k = 1;
            options.WeightMode = 'Cosine';
            options.bNormalized = 1;
            options.num_iter = T;
            options.delta = 0.5;
            [acc_per_image{source_domain_index}{target_domain_index}{randseed},acc_per_class{source_domain_index}{target_domain_index}{randseed}] = HDA_CDLPP(trainS_features,trainS_labels,trainT_features,trainT_labels,test_features,test_labels,options);
            end
        end
    end
end
save(['./results_semi_HDA/textdata-' classifierType '-sp-' num2str(using_sp) '-LTS-' num2str(num_training_per_class_target) '-numTestPerClass-' num2str(num_test_per_class) '-alpha-' num2str(alpha) '-lambda-' num2str(lambda) '-deltaS-' num2str(deltaS) '-deltaT-' num2str(deltaT) '-randseed-' num2str(randseed) '-PcaDim-' num2str(pcaDim) '-LppDim-' num2str(lppDim) '-T-' num2str(T) '.mat'],'acc_per_class*','acc_per_image*');
%save('./text-dataSplits.mat', 'dataSplits', 'sourceDataSplits');