# An-Implementation-of-Gray-World-Algorithm
A simple implementation of gray world algorithm.
* 实现了基础版本grey_world_0，参数分别为输入的图片，输出地址，选择处理的模式
* 模式0代表用图像本身的RGB均值处理，模式1代表假设我们拥有的均值是128
* 为了探讨同一图片不同位置处理结果的差异，我们实现了grey_world_1函数，参数分别为输入的图片，输出地址，x轴切割块数，y轴切割块数，处理模式
* 文件格式：
  * 测试图片：./test_picture/
  * 输出地址：./output_picture/
    * ./output_picture/grey_world_0/ 代表使用grey_world_0函数
    * ./output_picture/grey_world_1/ 代表使用grey_world_1函数
* 运行命令：python grey_world.py
