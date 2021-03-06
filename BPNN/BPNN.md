# BPNN
## 概况
- 语言：python
- 引用库：numpy, pandas
- 文件
    - test.py：入口文件
    - BP.py：BPNN类
    - BP1.py：可变隐藏层数BPNN类
    - util.py：工具类
    - iris.txt：训练数据
    - test.txt：测试数据

## 实现功能
### BP.py文件
- 输出维度可变
- 输入维度可变
- 隐藏层节点可变
- learning rate可变
- 激活函数：logistic和tanh（其中任意一个）

### BP1.py
在BP.py的基础上，额外实现了隐藏层数可变

## 测试
- 数据来自[UCI](https://archive.ics.uci.edu/ml/datasets.html)，数据为3种不同的花的各种数据
- 数据总共150组，从中拿9组来预测，141组数据迭代1000次进行训练
- 测试目标：根据给定数据训练模型，输入预测数据预测花的种类
- 数字0.1  0.2  0.3分别代表不同种类的花，测试结果如下,与实际情况基本符合
![](http://wx4.sinaimg.cn/mw690/006AMixJly1fxqe0t1e1ej30n0075mxa.jpg)

## 总结
这次神经网络的学习是我第一次接触机器学习，感觉非常有趣，其中数学公式的理解比较困难。还有数组的操作比较麻烦，借助numpy库可大大降低代码复杂度另：BP1.py虽然实现了多层隐藏层，但效果反而没有BP.py的单层隐藏好，可能我对公式的理解有所偏差，暂时未找到其中的bug
