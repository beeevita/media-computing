% Project Title: Human Action Recognition
% Author: Manu B.N
% Contact: manubn88@gmail.com
% Main module of action recognition

% 逐帧读取视频
close all
clear all
clc

% 数据集目录
filePath = [pwd, '//'];
dir = dir(fullfile(filePath, '*.mp4'));
class_num = 20;  % 总类别数目
each_num = 100;  % 每个类别的视频数目
train_num = each_num * 0.8;

% delete('./Frames/*.jpg');

train_features = [];
train_labels = [];
test_features = [];
test_labels = [];
cnt = 1;
for class = 1:class_num
   for video = 1:each_num
       delete('./Frames_hmdb/*.jpg');
       video_id = (class-1)*each_num + video;
%        if video_id==4
%            continue;
%        end
       aname = sprintf('%s%d.mp4',filePath,video_id)

       I = VideoReader(aname);
       nFrames = I.numberofFrames;
       vidHeight =  I.Height;
       vidWidth =  I.Width;
       mov(1:nFrames) = ...
        struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
               'colormap', []);
       if nFrames-1 < 50
           WantedFrames = nFrames -1;
       else
           WantedFrames = 50;
       end
%        step = (nFrames-1)/WantedFrames
%        WantedFrames = min(50, nFrames-1);
       % 存储所取的帧的图片
       for k = 1:WantedFrames
            mov(k).cdata = read(I, k);
            mov(k).cdata = imresize(mov(k).cdata,[256,256]);
            imwrite(mov(k).cdata,['./Frames_hmdb/',num2str(k),'.jpg']);
       end
       clc
       for i = 1:WantedFrames
            img=imread(['./Frames_hmdb/',num2str(i),'.jpg']);
            f1=il_rgb2gray(double(img));  % 将图片转换为一个通道上
            [ysize,xsize]=size(f1);  % shape
            nptsmax=40;   
            kparam=0.04;  
            pointtype=1;  
            sxl2=4;       
            sxi2=2*sxl2;  
            % detect points
            % 对于每一帧 提取STIP特征
            [posinit,valinit]=STIP(f1,kparam,sxl2,sxi2,pointtype,nptsmax);
            Test_Feat(i,1:40)=valinit;
       end
       if do_bow
           desc = struct('features',Test_Feat);
           if video <= train_num
               train_features = [train_features;desc];
               train_labels = [train_labels;class];
           else
               test_features = [test_features;desc];
               test_labels = [test_labels;class];
           end
       else
           Test_Feat = mean(Test_Feat); % 取平均值
           if video <= train_num
               train_features = [train_features;Test_Feat];
               train_labels = [train_labels;Test_Feat];
           else
               test_features = [test_features;Test_Feat];
               test_labels = [test_labels;Test_Feat];
           end 
       end
       cnt = cnt+1;
       
   end
end
if do_bow
    save hmdb_bow_train_features.mat train_features
    save hmdb_bow_train_labels.mat train_labels
    save hmdb_bow_test_features.mat test_features
    save hmdb_bow_test_labels.mat test_labels
else
    save hmdb_train_features.mat train_features
    save hmdb_train_labels.mat train_labels
    save hmdb_test_features.mat test_features
    save hmdb_test_labels.mat test_labels
end


