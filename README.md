# 简介
hahaha~~ 很开心又开始做新项目了，是《数字图像分析》大作业。实现文档提取与矫正。
*整个项目只用到了opencv*的IO操作(包括卷积，生成高斯滤波器等等) 

# 更新
- 2020.07.10
    - 使用 [NumCpp](https://github.com/dpilger26/NumCpp) / [pybind11](https://github.com/pybind/pybind11) 与C++混合编程, 加速卷积过程
    ![affQLq.png](https://s1.ax1x.com/2020/08/07/affQLq.png)
- 2020.07.07
    - 使用im2col代替原来三重for循环的卷积形式，整体运行时间能减少一半
    [![UFaXB8.png](https://s1.ax1x.com/2020/07/07/UFaXB8.png)](https://imgchr.com/i/UFaXB8)
    

# 运行纯python
> 默认. 有时间我添加下参数选项, 选择python还是c++版本

## 环境
- `pip install opencv-python numpy`
## run
- `python main.py`
- 在`main.py`中可以修改需要提取的文件，结果保存在`result`文件夹中

# 运行cpp混编
> 非默认. 有时间我添加下参数选项, 选择python还是c++版本

- 配置好[NumCpp](https://github.com/dpilger26/NumCpp) / [pybind11](https://github.com/pybind/pybind11)
- 运行`compile.sh`脚本
- 在`corner_detection.py`文件中修改c++版本卷积

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

# 结果
![000026.jpg](https://i.loli.net/2019/01/05/5c2ffcc192ae4.jpg)
![000026.jpg](https://i.loli.net/2019/01/05/5c2ffcfe5021c.jpg)
![000872.jpg](https://i.loli.net/2019/01/05/5c2ffcc194ee2.jpg)
![000872.jpg](https://i.loli.net/2019/01/05/5c2ffcfe4e4c4.jpg)
![001402.jpg](https://i.loli.net/2019/01/05/5c2ffcc1a6844.jpg)
![001402.jpg](https://i.loli.net/2019/01/05/5c2ffcfe51d36.jpg)
![001552.jpg](https://i.loli.net/2019/01/05/5c2ffcc1a7a7a.jpg)
![001552.jpg](https://i.loli.net/2019/01/05/5c2ffcfe53f1c.jpg)
![001201.jpg](https://i.loli.net/2019/01/05/5c2ffcc1a78a6.jpg)
![001201.jpg](https://i.loli.net/2019/01/05/5c2ffcfe55b12.jpg)
