% % Project Title: Human Action Recognition
% % Author: Manu B.N
% % Contact: manubn88@gmail.com
% % Main module of action recognition
% 
% close all
% clear all
% clc
% delete('./KTH_vis_mosift/*.jpg');
% dataset = 'dtdb';
% 
% filePath = [pwd, '/KTH/'];
% dir = dir(fullfile(filePath, '*.mp4'));
% each_num = 100;
% n_class=16;
% for i = 1:each_num:each_num*n_class
%     aname = sprintf('%s%d.mp4',filePath,i);
%     I = VideoReader(aname);
% %     implay(aname);
% %     pause(3);
%     nFrames = I.numberofFrames;
%     vidHeight =  I.Height;
%     vidWidth =  I.Width;
%     mov(1:nFrames) = ...
%         struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
%                'colormap', []);
%            WantedFrames = min(50, nFrames);
%     for k = 1:WantedFrames
%         mov(k).cdata = read( I, k);
%         mov(k).cdata = imresize(mov(k).cdata,[256,256]);
%         imwrite(mov(k).cdata,['./KTH_vis_mosift/',num2str(k),'.jpg']);
%     end
% 
%     clc
%     for j=1:WantedFrames
%     %     disp(['Processing frame no.',num2str(i)]);
%       img=imread(['./KTH_vis_mosift/',num2str(j),'.jpg']);
%       f1=il_rgb2gray(double(img));
%       % 加载预提取特征
%       mosift = load(['../mosift/', dataset, '/',[num2str(i), '.mat']]);
%       if strcmp(dataset,'dtdb') || strcmp(dataset,'hmdb')
%           mosift = mosift.sift;
%       end
%       disp(['loading from ', '../mosift/', dataset, '/',[num2str(i), '.mat']]);
%       imshow(f1,[],'border','tight'), hold on
%       axis off;
%       pos = mosift(mosift(:,3)==j,:);  % 提取这一帧的mosift特征的位置
%       for k=1:size(pos,1)
%           plot(pos(k,1),pos(k,2),'ro');  % pos(k,1)表示x坐标，pos(k,2)表示y坐标
%       end
%       
%       F=getframe(gcf);
%       temp_path = ['./KTH_vis_mosift/',num2str(i),'/'];
%       imwrite(F.cdata,[temp_path,num2str(j),'.jpg']);
%       disp(['saving in',temp_path]);
%     end
% end
% disp('Down!');
% 
% Project Title: Human Action Recognition
% Author: Manu B.N
% Contact: manubn88@gmail.com
% Main module of action recognition

close all
clear all
clc
delete('./KTH_vis_mosift/*.jpg');
dataset = 'KTH';

filePath = [pwd, '/KTH/'];
dir = dir(fullfile(filePath, '*.mp4'));
each_num = 100;
n_class=6;
for i = 1:each_num:each_num*n_class
    aname = sprintf('%s%d.mp4',filePath,i);
    I = VideoReader(aname);
%     implay(aname);
%     pause(3);
    nFrames = I.numberofFrames;
    vidHeight =  I.Height;
    vidWidth =  I.Width;
    mov(1:nFrames) = ...
        struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
               'colormap', []);
           WantedFrames = min(50, nFrames);
    for k = 1:WantedFrames
        mov(k).cdata = read( I, k);
        % mov(k).cdata = imresize(mov(k).cdata,[256,256]);
        imwrite(mov(k).cdata,['./KTH_vis_mosift/',num2str(k),'.jpg']);
    end

    clc
    for j=1:WantedFrames
    %     disp(['Processing frame no.',num2str(i)]);
      img=imread(['./KTH_vis_mosift/',num2str(j),'.jpg']);
      f1=il_rgb2gray(double(img));
      mosift = load(['../mosift/', dataset, '/',[num2str(i), '.txt']]);
      % 加载预提取特征
      disp(['loading from ', '../mosift/', dataset, '/',[num2str(i), '.txt']]);
      imshow(f1,[],'border','tight'), hold on
      axis off;
      pos = mosift(mosift(:,3)==j,:);  % 提取这一帧的mosift位置
      for k=1:size(pos,1)
          plot(pos(k,1),pos(k,2),'ro');  % pos(k,1)表示x坐标，pos(k,2)表示y坐标
          quiver(pos(k,1), pos(k,2),pos(k,5), pos(k,6))  % 方向可视化
      end
      
      F=getframe(gcf);
      temp_path = ['./KTH_vis_mosift/',num2str(i),'/'];
      imwrite(F.cdata,[temp_path,num2str(j),'.jpg']);
      disp(['saving in',temp_path]);
    end
end
disp('Down!');

