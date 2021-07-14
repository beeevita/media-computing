route='./MoSIFT/KTH/';%基本路径
for dir=1:100:600
    route='./MoSIFT/KTH/';%基本路径
    mp4file = [route, num2str(dir), '_mosift', '.mp4'];
    route = [route, num2str(dir), '/'];
    WriterObj=VideoWriter(mp4file,'MPEG-4');%待合成的视频的文件路径
    open(WriterObj);

    %n_frames=numel(d);% n_frames表示图像帧的总数
    for j=1:35
    filename=strcat(route,num2str(j),'.jpg');
    frame=imread(filename);%读取图像，放在变量frame中
    
    writeVideo(WriterObj,frame);%将frame放到变量WriterObj中
    %%为每一帧图像编号
    end
    close(WriterObj);
end
