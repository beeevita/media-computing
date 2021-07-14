# 目录结构

```cpp
.
├── code
│   ├── extraction_stip.m    // 
│   ├── img2vedio.m   // 将特征点打到每一帧上
│   ├── recognize_mosift.m  // 将mosift特征点打到视频的每一个图片上
│   ├── recognize_stip.m
│   ├── val_mosift.m
│   └── val_stip.m
├── visualization
│   ├── MoSIFT
│   │   ├── KTH
│   │   ├── dtdb
│   │   └── hmdb
│   └── STIP
│       ├── KTH
│       ├── dtdb
│       └── hmdb
├──实验详细数据-混淆矩阵.xlsx    
└── 媒体计算实验报告-郭清妍.pdf
```

## code_submit

只包含主要的运行脚本。

### 可视化部分

####recognize_mosift.m 

将mosift特征点打到视频的每一帧中，同时可视化mosift梯度变化方向并且将图片存储。

#### recognize_stip.m

将stip特征点打到视频的每一帧中，然后将图片存储。

#### img2video.m

将上一步生成的图片合成一个视频。

### 实验部分

#### extraction_stip.m

提取STIP特征点，同时分割训练集和测试集。

* 需要设置数据集，以及数据集类别个数`n_class`、每一类的样本数`each_num`
* 可以根据需求调整训练集与测试集的比例。
* 可以设置是否使用BoW算法
  * 使用BoW：需要存储视频每一帧的数据
  * 不使用BoW：选取视频的前50帧，分别提取特征点，然后取平均值存储

* 将结果存储为`.mat`格式。

#### va l_stip.m

评估基于STIP特征提取对视频动作识别效果的主函数代码，可以设置以下超参数：

* 是否使用BoW算法
* 选择SVM核函数：RBF，线性，卡方，SVM最大迭代次数
* 选择不同的数据集：dtdb，hmdb51，KTH，以及数据集类别数目，每一类样本的数目
* 聚类算法类心个数
* 训练集测试集的比例（需要与提取特征的时候一致，因为训练集和测试集是分开保存的）
* 是否使用特征归一化
* codebook容量

#### va l_mosift.m

评估基于mosift特征提取对视频动作识别效果的主函数代码，可以设置以下超参数：

* 选择不同的数据集：dtdb，hmdb51，KTH，以及数据集类别数目，每一类样本的数目
* 是否随机选择特征，随机选择特征的百分比
* 选择前多少个特征构建codebook
* 选择SVM核函数：RBF，线性，卡方，SVM最大迭代次数
* 是否使用特征归一化
* codebook容量

## visualzation

存储dtdb，hmdb51，KTH三个数据集MoSIFT、STIP的特征可视化视频。



> 如有需要，所有运行过程中用到的scripts以及原本的avi数据集和抽取的mosift和stip特征，联系：qingyan@tju.edu.cn

