
# Sklearn 学习目录
<img src='https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano%20cover%20page.jpg?raw=true' height=200>

Theano 算得上是 Tensorflow 的前身, 也是一种非常优秀的神经网络模块. 他可以实现各种各样的神经网络模式. 而且用起来也很自由. 兼容Windows, MacOS, Linux.

* 这套教材主要包含了:
 * 了解 的使用规范和原则;
 * 知道怎么去找合适的机器学习方法;
 * 如何按照规范调整个个参数;
 * 运用它强大的数据库;
 * 机器学习的一些重要原则;
 
 
---
## *点击标题进入Youtube视频教程*

1. [**Why?**](https://www.youtube.com/watch?v=84yGQZE43OU&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=2)
  * 介绍 Theano.
  * 在这之前, [<什么是机器学习>](https://www.youtube.com/watch?v=YY7-VKXybjc&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin&index=1), [<什么是神经网络>](https://www.youtube.com/watch?v=RSRkp8VAavQ&index=2&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin)这两段简短的视频介绍都是很推荐的. 优酷的这两段视频在这: [<什么是机器学习>](http://v.youku.com/v_show/id_XMTYyMjk2NDIwOA==.html?f=27892935&o=1), [<什么是神经网络>](http://v.youku.com/v_show/id_XMTU5NDc3MDQwOA==.html?f=27892935&o=1).
  * [优酷链接](http://v.youku.com/v_show/id_XMTY1OTQ4NDI2OA==.html?f=27743371&o=1)


2. [**安装**](https://www.youtube.com/watch?v=uefJFOaypj8&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=3)
  * 介绍如何安装. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano2_install.py))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY1OTUyNjIzNg==.html?f=27743371&o=1) 


3. [**神经网络在什么**](https://www.youtube.com/watch?v=sPu4KpzLaDQ&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=4)
  * 用动画的形式可视化学习的过程. 这个例子我们会在之后的[回归例子](https://www.youtube.com/watch?v=EULCWeavwPU&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=10)中详细讲解. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano3_what_does_ML_do.py))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2MTU4MzM4MA==.html?f=27743371&o=1)


4. [**基本用法**](https://www.youtube.com/watch?v=je2oHCX5m74&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=5)
  * Theano 的基本使用. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano4_basic_usage.py))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2MTY1NDY1Ng==.html?f=27743371&o=1)


5. [**function 用法**](https://www.youtube.com/watch?v=je2oHCX5m74&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=6)
  * function 的基本用法. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano5_function.py))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2MjY5NTI5Ng==.html?f=27743371&o=1)


6. [**shared 变量**](https://www.youtube.com/watch?v=2exmT0L-QV0&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=7)
  * 用 shared 来控制 weights, biases 的参数变量. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano6_shared_variable.py))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2Mjc3NTU4NA==.html?f=27743371&o=1)


7. [**Activation function 激励函数**](https://www.youtube.com/watch?v=GbYWEOjjsAI&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=8)
  * 参考<机器学习-简介系列>的[激励函数4分钟介绍](). 优酷的[激励函数4分钟介绍在这里](). ([代码](https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano7_activation_function.py))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2MzkxNDE1Ng==.html?f=27743371&o=1)


8. [**定义 Layer 类**](https://www.youtube.com/watch?v=Xm2InCJqFY4&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=9)
  * 定义Layer这个class, 便于之后添加神经层. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano8_Layer_class.py))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2Mzk3MDI2MA==.html?f=27743371&o=1)


9. [**Regression 回归例子**](https://www.youtube.com/watch?v=lWvlKqvvXyw&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=10)
  * 一个简单的线性回归例子. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/theanoTUT/theano9_regression_nn))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2NDE2MjA5Ng==.html?f=27743371&o=1)


10. [**可视化回归例子**](https://www.youtube.com/watch?v=EULCWeavwPU&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=11)
  * 我们可视化这个神经网络的学习过程. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/theanoTUT/theano10_regression_visualization))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2NDE5MDY2NA==.html?f=27743371&o=1)


11. [**Classification 分类例子**](https://www.youtube.com/watch?v=nslbfsN8wiU&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=12)
  * 一个简单的分类例子. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/theanoTUT/theano11_classification_nn))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2NDI3ODc2NA==.html?f=27743371&o=1)


12. [**Regularization 正规化**](https://www.youtube.com/watch?v=ho4ms9gVjKE&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=11)
  * 请参考<机器学习-简介系列>的[正规化简介](https://www.youtube.com/watch?v=e9OKufD6lRM&list=PLXO45tsB95cIFm8Y8vMkNNPPXAtYXwKin&index=10)>的4分钟介绍. (优酷的[正规化简介](http://v.youku.com/v_show/id_XMTczNjA2Nzc5Ng==.html?f=27892935&o=1)). ([代码](https://github.com/MorvanZhou/tutorials/tree/master/theanoTUT/theano12_regularization))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2NTAwNTk0MA==.html?f=27743371&o=1)


13. [**保存 model**](https://www.youtube.com/watch?v=sj9BGXGyLho&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=14)
  * 学习好了 model, 我们也要保存学好的参数. ([代码](https://github.com/MorvanZhou/tutorials/tree/master/theanoTUT/theano13_save))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2NTAyNTM0MA==.html?f=27743371&o=1)


14. [**总结和更多**](https://www.youtube.com/watch?v=2VzuMu589MQ&list=PLXO45tsB95cKpDID642AjNkygrSR5X15T&index=15)
  * 总结之前的东西和之后可以继续学习的东西. ([代码](https://github.com/MorvanZhou/tutorials/blob/master/theanoTUT/theano14_summary.py))
  * [优酷链接](http://v.youku.com/v_show/id_XMTY2NTA0ODA5Mg==.html?f=27743371&o=1)
