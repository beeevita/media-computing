% Project Title: Human Action Recognition
% Author: Manu B.N
% Contact: manubn88@gmail.com
% Main module of action recognition

close all
clear all
clc
delete('./KTH_vis/*.jpg');

filePath = [pwd, '/KTH/'];
dir = dir(fullfile(filePath, '*.mp4'));
each_num = 100;
n_class=6;
for i = 401:each_num:each_num*n_class
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
           WantedFrames = 50;
    for k = 1:WantedFrames
        mov(k).cdata = read( I, k);
        mov(k).cdata = imresize(mov(k).cdata,[256,256]);
        imwrite(mov(k).cdata,['./KTH_vis/',num2str(k),'.jpg']);
    end

    clc
    for j=1:WantedFrames
    %     disp(['Processing frame no.',num2str(i)]);
      img=imread(['./KTH_vis/',num2str(j),'.jpg']);
      f1=il_rgb2gray(double(img));
      [ysize,xsize]=size(f1);
      nptsmax=40;   
      kparam=0.04;  
      pointtype=1;  
      sxl2=4;       
      sxi2=2*sxl2;  
      % detect points
      [posinit,valinit]=STIP(f1,kparam,sxl2,sxi2,pointtype,nptsmax);
      Test_Feat(j,1:40)=valinit;

      imshow(f1,[]), hold on
      axis off;
      showellipticfeatures(posinit,[1 1 0]);
      % title('Feature Points','fontsize',12,'fontname','Times New Roman','color','Black')
      F=getframe(gcf);
      temp_path = ['./KTH_vis/',num2str(i),'/'];
      imwrite(F.cdata,[temp_path,num2str(j),'.jpg']);
      disp(['saving in',temp_path]);
    end

    % Use KNN To classify the videos
    % load('TrainFeat.mat');
%     load('X_trainFeat.mat')
%     load('Y_train.mat')
%     X = X_trainFeat;
%     Y = Y_train;
%     Z = mean(Test_Feat);
    % Now Classify

    %ens = fitensemble(X,Y,'Subspace',300,'KNN');
    %class = predict(ens,Z(1,:))
%     md1 = ClassificationKNN.fit(X,Y);
%     Type = predict(md1,Z);
    % Actions = ['boxing' 'handclapping' 'handwaving' 'jogging' 'running' 'walking'];
    %Actions = char('boxing','handclapping','handwaving','jogging','running','walking');

    % œ‘ æ¿‡±
%     info = Actions(Type, :);
%     disp(info);
%     helpdlg(info);
end
disp('Down!');

