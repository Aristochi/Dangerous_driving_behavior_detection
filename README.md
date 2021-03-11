运行环境：

1.python 3.7.4 2.pytorch 1.4.0
3.python-opencv

说明
预训练的权重文件[vgg_16] 具体的配置文件请看Config.py文件
训练运行python Train.py
单张测试 python Test.py

##目前进度： 1、PERCLOS计算 DONE 2、眨眼频率计算 DONE 3、打哈欠检测及计算 DONE 4、疲劳检测 DONE 5、人脸情绪检测 DONE 6、口罩检测Done

网络检测性能：准确率82.18%

主要文件说明： ssd_net_vgg.py 定义class SSD的文件 Train.py 训练代码 voc0712.py 数据集处理代码（没有改文件名，改的话还要改其他代码） loss_function.py 损失函数 detection.py 检测结果的处理代码，将SSD返回结果处理为opencv可以处理的形式 test.py 单张图片测试代码 Ps:没写参数接口，所以要改测试的图片就要手动改代码内部文件名了 l2norm.py l2正则化 Config.py 配置参数 utils.py 工具类 camera_detection.py 摄像头检测代码V1,V2 augmentations.py 生成识别框 Run.py 主程序运行文件 MainWindow.py UI界面布局 /Emoji 人脸情绪识别部分 /mtcnn 人脸定位检测 /test 测试视频、图片素材 /Facemask口罩检测相关结构

数据集结构： /dataset: /Annotations 存放含有目标信息的xml文件 /ImageSets/Main 存放图片名的文件 /JPEGImages 存放图片 /txt.py 生成ImageSets文件的代码

权重文件存放路径： /weights
数据集
链接：https://pan.baidu.com/s/17OT61-ZOgCTuceDBJa36eA 
提取码：2l7e 
复制这段内容后打开百度网盘手机App，操作更方便哦
带训练和测试代码版本请看https://github.com/DohaerisT/DangerousDrivingDetector
参考资料： https://blog.csdn.net/zxd52csx/article/details/82795104
