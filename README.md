# arcface-Paddle
基于Paddle框架的arcface复现

## ArcFace-Paddle

本项目基于paddlepaddle框架复现ArcFace，并参加百度第三届论文复现赛，将在2021年5月15日比赛完后提供AIStudio链接～敬请期待

参考项目：

[InsightFace](https://github.com/deepinsight/insightface)

## Paddle版本：
paddlepaddle-gpu==2.0.2

## 数据集
[MS1M-ArcFace](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
解压数据集，你应该得到以下目录结构
```
faces_more
|───property
└───cplfw.bin
└───agedb_30.bin
└───vgg2_fp.bin
└───lfw.bin
└───cfp_ff.bin
└───cfp_fp.bin
└───calfw.bin
└───train.rec
└───train.idx
```
**其中`train.rec`包含训练的图像，`train.idx`包含训练的标签，其均为`mxnet`数据格式，其余`.bin`文件均为二进制`bytes`文件**

## 训练
整个工程文件具有以下目录结构
```
|───faces_more
└───eval
└───mxnet_reader
└───mxnet_reader_win10
└───backbones
└───paddle_pretrainedmodel
└───utils
└───dataset.py
└───losses.py
└───partial_fc.py
└───config.py
└───train.py
```
**注意：`mxnet_reader`用于`Linux`系统部署训练，`mxnet_reader_win10`用于win10系统部署训练，两者均为重构`mxnet`数据读取后的代码**
### 配置说明
`config.py`里面包含训练的超参数，学习率衰减函数，训练文件路径以及验证文件列表

`backbones`里面包含提供的训练模型，`iresnet18`、`iresnet34`、`iresnet50`、`iresnet100`、`iresnet200`

`partial_fc`来源于论文《Partial FC: Training 10 Million Identities on a Single Machine》，其目的是加速训练超大规模数据集

`paddle_pretrainedmodel`包含网络的预训练文件，其均为由torch模型转换而来，里面包含测试代码`model_test.py`以及精度文件`results.txt`

### 启动训练
```python
python train.py [--network XXX]
``` 
**这将会在log文件夹下产生训练的日志文件，其包括损失值以及所需训练的的时间，工程中的`training.log`包含了部分训练过程中的打印信息**

![](https://ai-studio-static-online.cdn.bcebos.com/92a05a4dcc5f4d00a08c1ee6bfcfbcfeb345c72263454d5a9aa65e49c5aa895f)

**训练过程中的权重文件将保存在`emore_arcface_r50`文件夹下，保存路径源于你的`config`文件设置，你应具有以下类似目录**
```
|───emore_arcface_r50
└───backbone.pdparams
└───rank:0_softmax_weight.pkl
└───rank:0_softmax_weight_mom.pkl
```
本次利用`aistudio`训练的`iresnet50`得到的`backbone.pdparams`精度如下，其中`lfw=0.99750`，`cplfw=0.92117`，`calfw=0.96017`，你可以通过修改`/home/aistudio/paddle_pretrainedmodel/ model_test.py`权重路径`model_params=/home/aistudio/emore_arcface_r50/backbone.pdparams`来测试自己的模型

![](https://ai-studio-static-online.cdn.bcebos.com/ebcbdb1ff96845ba9e2174434807b1cbae3d6293a0444110b73f9e827b403280)

由于aistudio对保存版本文件的限制，我将保存的文件已上传至我的服务器，你可以通过`wget ftp://207.246.98.85/emore_arcface_r50.zip`下载获取

<img src="https://ai-studio-static-online.cdn.bcebos.com/5229b50aa5c74ca1937f50ee4685ac5d69ff12bbed604a1a9ae77a33cc90e1d9" width="700"/>

### 启动测试

[模型和数据集读取代码下载](https://pan.baidu.com/s/14rZuYMfvO9RIHZmPhwoUqQ)

提取码：dzc0 

[AIStudio链接](https://aistudio.baidu.com/aistudio/projectdetail/1770049?channel=0&channelType=0&shared=1)

```python
cd /home/aistudio/paddle_pretrainedmodel
python model_test.py [--network XXX]
```
**注意到`model_test.py`测试的官方提供的预训练模型，测试自己的训练模型，你需要修改读取文件的路径以及网络结构**

## **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| 主页        | [Deep Hao的主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
