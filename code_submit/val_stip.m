load('hmdb_train_features.mat');
load('hmdb_train_labels.mat');
load('hmdb_test_features.mat');
load('hmdb_test_labels.mat');
do_svm_rbf_classification= 0;
do_svm_chi2_classification = 0;
do_svm_linar_classification = 1;
do_bow = 0;
nwords_codebook = 64;
norm_bof_hist = 1;
n_class = 20;
each_num = 100;
each_train_num = each_num * 0.8;
each_test_num = each_num-each_train_num;


% BOW PARAMETERS
max_km_iters = 50; % maximum number of iterations for k-means
nfeat_codebook = 60000; % number of descriptors used by k-means for the codebook generation
norm_bof_hist = 1;

X_train = train_features;
X_test = test_features;
Y_train = train_labels;
Y_test = test_labels;
data_X = [train_features;test_features];
data_Y = [train_labels;test_labels];
[train_num, o] = size(train_labels);
[nsample,nfeatures] = size(data_X);

%% BOW  
% ����ѵ���������Լ� ������STIP����
if do_bow 
    lasti=1;
    for i = 1:n_class
         for j = 1:each_train_num
             if lasti > train_num
                 break
             end
            desc_train(lasti)=struct('features',X_train(lasti).features,'class', Y_train(lasti));
            desc_train(lasti).features = single(desc_train(lasti).features);
            lasti=lasti+1;
        end;

        if lasti > train_num
            break
        end
    end;

    lasti=1;
    for i = 1:n_class
         for j = 1:each_test_num
            desc_test(lasti)=struct('features',X_test(lasti).features,'class', Y_test(lasti));
            desc_test(lasti).features = single(desc_test(lasti).features);
            lasti=lasti+1;
        end;
    end;

    %% build codebook
    fprintf('\nBuild visual vocabulary:\n');
    % ��������Ƶ������������ȫ��ƴ������
    DESC = [];
    labels_train = cat(1,desc_train.class);
    for i=1:n_class
        desc_class = desc_train(labels_train==i);
        [w,h]=  size(desc_class);
        randimages = randperm(h);
        randimages =randimages(1:5);
        DESC = vertcat(DESC,desc_class(randimages).features);
        %display(size(DESC));
    end
    display(size(DESC,1));
    % ��ѵ������������������� ����codebook
    r = randperm(size(DESC,1));
    r = r(1:min(length(r),nfeat_codebook));

    DESC = DESC(r,:);

    % run k-means
    K = nwords_codebook; % size of visual vocabulary
    fprintf('running k-means clustering of %d points into %d clusters...\n',...
        size(DESC,1),K)
    % input matrix needs to be transposed as the k-means function expects 
    % one point per column rather than per row

    cluster_options.maxiters = max_km_iters;  % ����������
    cluster_options.verbose  = 1;

    [VC] = kmeans_bo(double(DESC),K,max_km_iters);%visual codebook
    VC = VC';%transpose for compatibility with following functions
    clear DESC;


    %% feature quantization
    fprintf('\nFeature quantization (hard-assignment)...\n');
    % ѵ����
    for i=1:length(desc_train)  
      features = desc_train(i).features(:,:);
      dmat = eucliddist(features,VC);  % ���� SIFT��������VC������������ŷʽ����
      [quantdist,visword] = min(dmat,[],2);  % ��VC�о�������Ĵ�ƽ��ֵ��������һ��sift������
      % save feature labels
      desc_train(i).visword = visword;  % ���������visual word
      desc_train(i).quantdist = quantdist;  % ??�??��?�?
    end

    % ���Լ�
    for i=1:length(desc_test)    
      features = desc_test(i).features(:,:); 
      dmat = eucliddist(features,VC);
      [quantdist,visword] = min(dmat,[],2);
      % save feature labels
      desc_test(i).visword = visword;
      desc_test(i).quantdist = quantdist;
    end

    % ѵ����
%     for i=1:length(X_train)  
%       feature = X_train(i);
%       dmat = eucliddist(feature,VC);  % ������һfeature map��VC������������ŷʽ����
%       [quantdist,visword] = min(dmat,[],2);  % ��VC�о�������Ĵ�ƽ��ֵ��������һ��feature map
%       % save feature labels
%       desc_train(i).visword = visword;  % ���������visual word
%       desc_train(i).quantdist = quantdist; 
%     end
% 
%     % ���Լ�
%     for i=1:length(X_test)    
%       feature = X_test(i); 
%       dmat = eucliddist(feature,VC);
%       [quantdist,visword] = min(dmat,[],2);
%       % save feature labels
%       desc_test(i).visword = visword;
%       desc_test(i).quantdist = quantdist;
%     end

    %% Represent each image by the normalized histogram of visual
    % word labels of its features. Compute word histogram H over 
    % the whole image, normalize histograms w.r.t. L1-norm.

    N = size(VC,1); % visual words�ĸ���

    for i=1:length(desc_train) 
        visword = desc_train(i).visword;  % ÿ��ͼƬ��Ӧ��visual word
        H = histc(visword,[1:nwords_codebook]);

        % L1������һ��
        if norm_bof_hist
            H = H/sum(H);
        end

        % save histograms
        desc_train(i).bof=H(:)';
    end

    for i=1:length(desc_test) 
        visword = desc_test(i).visword;
        H = histc(visword,[1:nwords_codebook]);

        % normalize bow-hist (L1 norm)
        if norm_bof_hist
            H = H/sum(H);
        end

        % save histograms
        desc_test(i).bof=H(:)';
    end

    X_train = cat(1,desc_train.bof);
    X_test = cat(1, desc_test.bof);
end


%% SVM classification
basepath = '..';
wdir = pwd;
libsvmpath = [ wdir(1:end-4) fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

%% RBF kernal function

if do_svm_rbf_classification
    % cross-validation
    C_vals=log2space(2,10,5);
    Gamma_vals = log2space(-5, 10, 10);

    % ѡ����õ�c��gammaֵ
    Acc_best = 0;
    for i=1:length(C_vals)
        for j=1:length(Gamma_vals)
            % -t=2�� ��ʾѡ������RBF����
            % -c����ʾ�ͷ�ϵ��
            opt_string=['-t 2 -v 5 -c ' num2str(C_vals(i)) ' -g ' num2str(Gamma_vals(j))];
            xval_acc=svmtrain(Y_train,X_train,opt_string);
            if xval_acc > Acc_best
                Acc_best = xval_acc;
                C_best = C_vals(i);
                Gamma_best = Gamma_vals(j);
            end
        end
    end
    % ����õ�c��gammaֵѵ��SVM����Ԥ��
    model=svmtrain(Y_train,X_train,['-t 2 -c ' num2str(C_best) ' -g ' num2str(Gamma_best)] );

    % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros.... 
    % consider this if the kernel is computationally inefficient.
    disp('*** SVM - RBF Gaussian kernel ***');
    [precomp_rbf_svm_lab,conf]=svmpredict(Y_test, X_test, model);
    acc = sum(Y_test==precomp_rbf_svm_lab)/length(Y_test);
    % confusion matrix
    CM = confmatrix(Y_test,precomp_rbf_svm_lab,n_class);
    method_name='SVM RBF Gaussian';
    % Compute classification accuracy
    fprintf('OVERALL %s classification accuracy: %1.4f\n\n',method_name,acc);
end

%% chi2 kernal function
if do_svm_chi2_classification    
    % compute kernel matrix
    Ktrain = kernel_expchi2(X_train,X_train);
    Ktest = kernel_expchi2(X_test,X_train);
    
    % cross-validation
    C_vals=log2space(2,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(Y_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(Y_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
    % we supply the missing scalar product (actually the values of non-support vectors could be left as zeros.... 
    % consider this if the kernel is computationally inefficient.
    disp('*** SVM - Chi2 kernel ***');
    [precomp_chi2_svm_lab,conf]=svmpredict(Y_test,[(1:size(Ktest,1))' Ktest],model);
    acc = sum(Y_test==precomp_chi2_svm_lab)/length(Y_test);
    % confusion matrix
    CM = confmatrix(Y_test,precomp_chi2_svm_lab,n_class);
    
    method_name='SVM Chi2';
    % Compute classification accuracy
    fprintf('OVERALL %s classification accuracy: %1.4f\n\n',method_name,acc);
end

%% SVM classification linear kernal (using libsvm) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if do_svm_linar_classification
    % cross-validation
    C_vals=log2space(7,10,5);  % �����ռ��趨
    for i=1:length(C_vals);
        % �����ռ����� ���۽�����֤
        opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(Y_train,X_train,opt_string);
    end
    % ѡ����õ�c
    [v,ind]=max(xval_acc);  % indΪC_vals�����cֵ������

    model=svmtrain(Y_train,X_train,['-t 0 -c ' num2str(C_vals(ind))]); % ѵ��
    disp('*** SVM - linear ***');
    svm_lab=svmpredict(Y_test,X_test,model);  % Ԥ��
    CM = confmatrix(Y_test,svm_lab,n_class); % ��������
    acc = sum(Y_test==svm_lab)/length(Y_test);  % ׼ȷ��
    method_name='SVM linear';
    fprintf('OVERALL %s classification accuracy: %1.4f\n\n',method_name,acc);
end

%% KNN classification
% Now Classify
%ens = fitensemble(X,Y,'Subspace',300,'KNN');
%class = predict(ens,Z(1,:))
% md1 = ClassificationKNN.fit(X_train,Y_train);
% Type = predict(md1,X_test)
% % ����׼ȷ��
% acc = sum(Y_test==Type)/length(Y_test);
% display(acc);
