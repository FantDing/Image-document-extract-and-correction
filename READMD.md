# 简介
hahaha~~ 很开心又开始做新项目了，是《数字图像分析》大作业。实现文档提取与矫正。
*整个项目只用到了opencv*的IO操作(包括卷积，生成高斯滤波器等等) 

# 运行
## 环境
- `pip install opencv-python numpy`
## run
- `python main.py`
- 在`main.py`中可以修改需要提取的文件，结果保存在`result`文件夹中

# 技术栈
## S&G ?
SG是自己乱起的，包含了两个技术栈：
- S： Susan角点检测
- G： Geometic几何校正

## H&G !
> 通过实验发现直接检测角点根本实现不了，图片中角点太多，且不能加入先验进行过滤。遂通过
检测直线，求角点实现 

- H: Hough哈夫变换检测直线，得到角点
- G: Geometic几何校正


