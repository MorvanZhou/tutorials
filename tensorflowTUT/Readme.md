# Tensorflow 视频教程结构

1. [Why? ](https://www.youtube.com/watch?v=vZ263nfbh8g&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=2)
... Tensorflow 简介
... [优酷链接](http://v.youku.com/v_show/id_XMTYxMzQzMDA3Mg==.html?f=27327189&o=1)

2. [安装](https://www.youtube.com/watch?v=pk6sAg2M-fU&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=3)
...介绍如何安装和安装时的限制
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMzQzMjEyNA==.html?f=27327189&o=1)

3. [例子1](https://www.youtube.com/watch?v=tM4z02cDNa4&index=4&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)
...用一个线性回归的例子来说明神经网络究竟在干什么. 我们还可视化了整个学习的过程. 代码和实现我们会在[例子3](https://www.youtube.com/watch?v=FTR36h-LKcY&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=11)中慢慢说.
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMzQzNDc5Ng==.html?f=27327189&from=y1.2-3.4.4&spm=a2h0j.8191423.item_XMTYxMzQzNDc5Ng==.A)

4. [处理结构](https://www.youtube.com/watch?v=9l_c5260JQ8&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=5)
...Tensorflow 的处理,代码结构可能和我们想象得不一样. 我们需要先定义好整个 graph, 也就是神经网络的框架,才能开始运算.
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMzQ1NzUwOA==.html?f=27327189&o=1)

5. [例子2](https://www.youtube.com/watch?v=JKR1Dxinwwc&index=6&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)
...这个例子是我们第一个开始将代码的例子. 我们来熟悉一下 tf 的代码吧. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf5_example2))
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMzQ2NzE0OA==.html?f=27327189&o=1)

6. [Session 会话控制](https://www.youtube.com/watch?v=HhjtJ73AwIY&index=7&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)
...Session 是 tf 的主要结构之一, 他骑着控制整个运算结构的功能. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow6_session.py))
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMzYzNTc2OA==.html?f=27327189&o=1)

7. [Variable 变量](https://www.youtube.com/watch?v=jGxK7gfglrI&index=8&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)
...我们会把 weights 还有 biases 当做变量来储存, 更新. 这个是介绍 variable 的基本使用方法. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow7_variable.py))
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMzY2MDM2OA==.html?f=27327189&o=1)

8. [Placeholder 传入值](https://www.youtube.com/watch?v=fCWbRboJ4Rs&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=9)
...定义好的神经网络结构, 我们就能用 placeholder 当作数据的接收口, 一次次传入数据. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow8_feeds.py))
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMzY5NzI4MA==.html?f=27327189&o=1)

9. [激励函数](https://www.youtube.com/watch?v=6gbGCxBGxZA&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=10)
...请参考激励函数在<[机器学习-简介系列](https://www.youtube.com/watch?v=tI9AbaBfnPc&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin&index=9)>里的4分钟介绍. ([优酷的<机器学习-简介系列>](http://v.youku.com/v_show/id_XMTcxMTExNjA5Mg==.html?f=27892935&o=1))
...[优酷链接](http://v.youku.com/v_show/id_XMTU5NjA2MTk0MA==.html?f=27327189&o=1)

10. [例子3 添加神经层](https://www.youtube.com/watch?v=FTR36h-LKcY&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=11)
...用简单的函数做一个添加神经层的功能, 以后再不断套用这个功能 ([代码](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tensorflow10_def_add_layer.py))
...[优酷链接](http://v.youku.com/v_show/id_XMTU5NjEzOTA4NA==.html?f=27327189&o=1)

11. [例子3 建造神经网络](https://www.youtube.com/watch?v=S9wBMi2B4Ss&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=12)
...运用上上次的添加层功能, 开始搭建神经网络. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf11_build_network))
...[优酷链接](http://v.youku.com/v_show/id_XMTU5OTA5NDI1Mg==.html?f=27327189&o=1)

12. [例子3 结果可视化](https://www.youtube.com/watch?v=nhn8B0pM9ls&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=13)
...对于怎个例子3的学习步骤的可视化教程. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf12_plot_result))
...[优酷链接](http://v.youku.com/v_show/id_XMTU5OTQzOTMzNg==.html?f=27327189&o=1)

13. [Optimizer 优化器](https://www.youtube.com/watch?v=9BmaWixFwj8&index=14&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)
...请参考优化器在<[机器学习-简介系列](https://www.youtube.com/watch?v=UlUGGB7akfE&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin&index=11)>里的4分钟介绍. ([优酷的<机器学习-简介系列>](http://v.youku.com/v_show/id_XMTc2MjA0ODQyOA==.html?f=27892935&o=1))
...[优酷链接](http://v.youku.com/v_show/id_XMTYwMzk1NDM4OA==.html?f=27327189&o=1)

14. [Tensorboard1 可视化好帮手](https://www.youtube.com/watch?v=SDeQRRRMUHU&index=15&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)
...神经网络结构和参数, 数据的可视化. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf14_tensorboard))
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMTYwMjEwMA==.html?f=27327189&o=1)

15. [Tensorboard2 可视化好帮手](https://www.youtube.com/watch?v=L-RDrbYNWDk&index=16&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)
...同上. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf15_tensorboard))
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMTcxODYyMA==.html?f=27327189&o=1)

16. [Classification 分类神经网络](https://www.youtube.com/watch?v=aNjdw9w_Qyc&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=17)
...搭建一个用于分类的神经网络. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf16_classification))
...[优酷链接](http://v.youku.com/v_show/id_XMTYxMjQ2NTYyNA==.html?f=27327189&o=1)

17. [Dropout 过拟合问题](https://www.youtube.com/watch?v=f2F9Xsd7KVk&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=18)
...请参考过拟合在<[机器学习-简介系列](https://www.youtube.com/watch?v=e9OKufD6lRM&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin&index=10)>的4分钟介绍. ([优酷的<机器学习-简介系列>](http://v.youku.com/v_show/id_XMTczNjA2Nzc5Ng==.html?f=27892935&o=1)). 这节实现了用 dropout 解决过拟合的途径. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf17_dropout))
...[优酷链接](http://v.youku.com/v_show/id_XMTYxODI2Mzk5Ng==.html?f=27327189&o=1)

18. [CNN 1 卷积神经网络](https://www.youtube.com/watch?v=tjcgL5RIdTM&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=19)
...请参考CNN 在<[机器学习-简介系列](https://www.youtube.com/watch?v=hMIZ85t9r9A&index=3&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin)>中的介绍. ([优酷的<机器学习-简介系列>](http://v.youku.com/v_show/id_XMTY4MzAyNTc4NA==.html?f=27892935&o=1))
...[优酷链接](http://v.youku.com/v_show/id_XMTYyMTUyMjc0OA==.html?f=27327189&o=1)

19. [CNN 2 卷积神经网络](https://www.youtube.com/watch?v=JCBe_yjDmY8&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=20)
...代码部分. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf18_CNN2))
...[优酷链接](http://v.youku.com/v_show/id_XMTYyMTY1MjMwOA==.html?f=27327189&o=1)

20. [CNN 3 卷积神经网络](https://www.youtube.com/watch?v=pjjH2dGGwwY&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=21)
...代码部分. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf18_CNN3))
...[优酷链接](http://v.youku.com/v_show/id_XMTYyMTc3ODc0OA==.html?f=27327189&o=1)

21. [Saver 保存参数](https://www.youtube.com/watch?v=R-22pnDezHU&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=22)
...训练好了以后, 我们可以保存这些 weights 和 biases 的参数,避免重复训练. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf19_saver.py))
...[优酷链接](http://v.youku.com/v_show/id_XMTYyNzE2MDUwOA==.html?f=27327189&o=1)

22. [RNN 循环神经网络](https://www.youtube.com/watch?v=i-cd3wzsHtw&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=23)
...请参考RNN 在<[机器学习-简介系列](https://www.youtube.com/watch?v=EEtf4kNsk7Q&index=4&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin)>中的4分钟介绍. 还有LSTM 在<[机器学习-简介系列](https://www.youtube.com/watch?v=Vdg5zlZAXnU&index=5&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin)>中的介绍.
优酷的这两段简介视频在这: [RNN简介](http://v.youku.com/v_show/id_XMTcyNzYwNjU1Ng==.html?f=27892935&o=1), [LSTM简介](http://v.youku.com/v_show/id_XMTc0MzY5MTQxMg==.html?f=27892935&o=1)
...[优酷链接](http://v.youku.com/v_show/id_XMTcyNjE0ODM4MA==.html?f=27327189&o=1)

23. [RNN LSTM 例子1 分类](https://www.youtube.com/watch?v=IASyrQamTQk&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=24)
...使用LSTM RNN 做 MNIST 图片集的分类问题. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf20_RNN2))
...[优酷链接](http://v.youku.com/v_show/id_XMTcyNjE5ODU3Mg==.html?f=27327189&o=1)

24. [RNN LSTM 例子2 回归](https://www.youtube.com/watch?v=nMLPYT_SMRo&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=25)
...使用LSTM RNN 做 sin, cos 曲线的回归问题. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf20_RNN2.2))
...[优酷链接](http://v.youku.com/v_show/id_XMTczMDY5Mjc5Ng==.html?f=27327189&o=1)

25. [RNN LSTM 例子2 回归的学习过程可视化](https://www.youtube.com/watch?v=V-pvtUThhNE&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=26)
...对于上面的例子的训练可视化. (代码同上)
...[优酷链接](http://v.youku.com/v_show/id_XMTczMDcxMjEwNA==.html?f=27327189&o=1)
