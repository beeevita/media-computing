%% parameters
basepath = '..';
dataset_dir = 'KTH';
nfeat_codebook = 60000; 
nwords_codebook = 64;
max_km_iters = 50;
norm_bof_hist = 1;
do_svm_rbf_classification= 1;
do_svm_chi2_classification = 1;
do_svm_linar_classification = 1;
do_knn = 0;
n_frame=1000000;  % 选取前n_frame个帧
do_rand = 1;  % 是否随机选取特征，
ratio = 0.2;  % 随机选取特征的比例

if strcmp('KTH', dataset_dir)
    n_class = 6;
    each_num = 100;
    ext = 'txt';
    dotext = '.txt';
elseif strcmp('hmdb51', dataset_dir)
    n_class = 20;
    each_num = 100;
    ext = '-mat';
    dotext = '.mat';
else
    n_class = 18;
    each_num = 100;
    ext = '-mat';
    dotext = '.mat';
end
each_train_num = each_num * 0.8;
each_test_num = each_num-each_train_num;
%% Load pre-computed features for training images

% The resulting structure array 'desc' will contain one
% entry per images with the following fields:
%  desc(i).r :    Nx1 array with y-coordinates for N SIFT features
%  desc(i).c :    Nx1 array with x-coordinates for N SIFT features
%  desc(i).rad :  Nx1 array with radius for N SIFT features
%  desc(i).sift : Nx128 array with N SIFT descriptors
%  desc(i).imgfname : file name of original image

lasti=1;
for i = 1:n_class
     for j = 1:each_train_num
        id =  (i-1)*100 + j;
        fname = fullfile(basepath,'mosift',dataset_dir,[num2str(id),dotext]);
        fprintf('Loading %s \n',fname);
        tmp = load(fname,ext);
        
        if strcmp(dataset_dir ,'hmdb51') || strcmp(dataset_dir ,'dtdb') 
            [w,h] = size(tmp.sift);
            tmp = tmp.sift;
        else
            [w,h] = size(tmp);
        end
        maxx = min(w, n_frame);
        tmp = tmp(:,7:end);
        if w==0
            display('w=0');
            continue
        end
        if do_rand
            tmp = tmp(randperm(floor(w*ratio)),:);
        else
            tmp = tmp(1:maxx,:);
        end
        
        desc_train(lasti)=struct('features',tmp, 'class',i);
        desc_train(lasti).features = single(desc_train(lasti).features);

        lasti=lasti+1;
     end
end

%% Load pre-computed SIFT features for test images 
test_id = 1;
for i = 1:n_class
     for j = 81:100
         id =  (i-1)*100 + j;
        fname = fullfile(basepath,'mosift',dataset_dir,[num2str(id),dotext]);
        fprintf('Loading %s \n',fname);
        tmp = load(fname,ext);
        
        if strcmp(dataset_dir ,'hmdb51') || strcmp(dataset_dir ,'dtdb') 
            [w,h] = size(tmp.sift);
            tmp = tmp.sift;
        else
            [w,h] = size(tmp);
        end
        if w==0
            display('w=0');
            continue
        end
        maxx = min(w, n_frame);
        tmp = tmp(:,7:end);   % 删掉前六列表示坐标的数据

        if do_rand  % 随机抽取
            tmp = tmp(randperm(floor(w*ratio)),:);
            size(tmp);
        else
            tmp = tmp(1:maxx,:);
        end
        
        desc_test(test_id)=struct('features',tmp, 'class',i);
        desc_test(test_id).features = single(desc_test(test_id).features);

        size(desc_test(test_id).features)
        test_id = test_id+1;
     end
end

%% Build visual vocabulary using k-means %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nBuild visual vocabulary:\n');

% concatenate all descriptors from all images into a n x d matrix 
DESC = [];
labels_train = cat(1,desc_train.class);
for i=1:n_class
    desc_class = desc_train(labels_train==i);
    [w,h]=  size(desc_class);
    randimages = randperm(h);
    %randimages =randimages(1:5);
    DESC = vertcat(DESC,desc_class(randimages).features);
end

% sample random M (e.g. M=20,000) descriptors from all training descriptors
r = randperm(size(DESC,1));
r = r(1:min(length(r),nfeat_codebook));

DESC = DESC(r,:);

% run k-means
K = nwords_codebook; % size of visual vocabulary
fprintf('running k-means clustering of %d points into %d clusters...\n',...
    size(DESC,1),K)
% input matrix needs to be transposed as the k-means function expects 
% one point per column rather than per row

% form options structure for clustering
cluster_options.maxiters = max_km_iters;
cluster_options.verbose  = 1;

[VC] = kmeans_bo(double(DESC),K,max_km_iters);%visual codebook
VC = VC';%transpose for compatibility with following functions
clear DESC;

%% K-means descriptor quantization means assignment of each feature
fprintf('\nFeature quantization (hard-assignment)...\n');
% 训练集
for i=1:length(desc_train)  
  features = desc_train(i).features(:,:); 
  dmat = eucliddist(features,VC);  % 计算 SIFT描述符与VC中所有向量的欧式距离
  [quantdist,visword] = min(dmat,[],2);  % 用VC中距离最近的簇平均值来量化这一个sift描述符
  % save feature labels
  desc_train(i).visword = visword;  % 距离最近的visual word
  desc_train(i).quantdist = quantdist;  
end

% 测试集
for i=1:length(desc_test)
  features = desc_test(i).features(:,:); 
  dmat = eucliddist(features,VC);
  [quantdist,visword] = min(dmat,[],2);
  % save feature labels
  desc_test(i).visword = visword;
  desc_test(i).quantdist = quantdist;
end

%% Represent each image by the normalized histogram of visual
N = size(VC,1); % number of visual words

for i=1:length(desc_train) 
    visword = desc_train(i).visword;
    H = histc(visword,[1:nwords_codebook]);
  
    % normalize bow-hist (L1 norm)
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

labels_train=cat(1,desc_train.class);
labels_test=cat(1,desc_test.class);
bof_train=cat(1,desc_train.bof);
bof_test=cat(1,desc_test.bof);

%% CHI-2 KERNEL (pre-compute kernel) %%%%%%%%%%%%%%%%%%%%%%%%%%%
basepath = '..';
wdir = pwd;
libsvmpath = [ wdir(1:end-4) fullfile('lib','libsvm-3.11','matlab')];
addpath(libsvmpath)

if do_svm_chi2_classification    
    % compute kernel matrix
    Ktrain = kernel_expchi2(bof_train,bof_train);
    Ktest = kernel_expchi2(bof_test,bof_train);
    
    % cross-validation
    C_vals=log2space(2,10,5);
    for i=1:length(C_vals);
        opt_string=['-t 4  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],opt_string);
    end
    [v,ind]=max(xval_acc);

    % train the model and test
    model=svmtrain(labels_train,[(1:size(Ktrain,1))' Ktrain],['-t 4 -c ' num2str(C_vals(ind))] );
    disp('*** SVM - Chi2 kernel ***');
    [precomp_chi2_svm_lab,conf]=svmpredict(labels_test,[(1:size(Ktest,1))' Ktest],model);
    acc = sum(labels_test==precomp_chi2_svm_lab)/length(labels_test);
    % confusion matrix
    CM = confmatrix(labels_test,precomp_chi2_svm_lab,n_class);
    method_name='SVM Chi2';
    fprintf('OVERALL %s classification accuracy: %1.4f\n\n',method_name,acc);

end
if do_svm_rbf_classification
    % cross-validation
    C_vals=log2space(2,10,5);
    Gamma_vals = log2space(-5, 10, 10);

    % 选择最好的c和gamma值
    Acc_best = 0;
    for i=1:length(C_vals);
        for j=1:length(Gamma_vals)
            % -t=2： 表示选择线性RBF函数
            % -c：表示惩罚系数
            opt_string=['-t 2 -v 5 -c ' num2str(C_vals(i)) ' -g ' num2str(Gamma_vals(j))];
            xval_acc=svmtrain(labels_train,bof_train,opt_string);
            if xval_acc > Acc_best
                Acc_best = xval_acc;
                C_best = C_vals(i);
                Gamma_best = Gamma_vals(j);
            end
        end
    end
    % 用最好的c和gamma值训练SVM并且预测
    model=svmtrain(labels_train,bof_train,['-t 2 -c ' num2str(C_best) ' -g ' num2str(Gamma_best)] );
    disp('*** SVM - RBF Gaussian kernel ***');
    [precomp_rbf_svm_lab,conf]=svmpredict(labels_test, bof_test, model);

    method_name='SVM RBF Gaussian';
    acc = sum(labels_test==precomp_rbf_svm_lab)/length(labels_test);
    % confusion matrix
    CM = confmatrix(labels_test,precomp_rbf_svm_lab,n_class);
    fprintf('OVERALL %s classification accuracy: %1.4f\n\n',method_name,acc);
end

% LINEAR SVM
if do_svm_linar_classification
    % cross-validation
    C_vals=log2space(7,10,5);  % 参数空间设定
    for i=1:length(C_vals);
        % 参数空间搜索 五折交叉验证
        opt_string=['-t 0  -v 5 -c ' num2str(C_vals(i))];
        xval_acc(i)=svmtrain(labels_train,bof_train,opt_string);
    end
    % 选择最好的c
    [v,ind]=max(xval_acc);  % ind为C_vals中最好c值的索引
    
    % train the model and test
    model=svmtrain(labels_train,bof_train,['-t 0 -c ' num2str(C_vals(ind))]);
    disp('*** SVM - linear ***');
    svm_lab=svmpredict(labels_test,bof_test,model);
    
    method_name='SVM linear';
    acc = sum(labels_test==svm_lab)/length(labels_test);
    % confusion matrix
    CM = confmatrix(labels_test,svm_lab,n_class);
    fprintf('OVERALL %s classification accuracy: %1.4f\n\n',method_name,acc);
end

if do_knn
    md1 = ClassificationKNN.fit(bof_train,labels_train);
    Type = predict(md1,bof_test)
    % 分类准确率
    acc = sum(labels_test==Type)/length(labels_test);
    display(acc);
end