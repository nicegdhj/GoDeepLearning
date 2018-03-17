这是我使用tensorflow进行深度学习的代码仓库

###my-tf-slim-demo


使用TF-Slim库快速上手CNN进行图片分类. 具体说明请参考我的个人博客文章<http://nicehjia.me/blog/2017/07/02/tensorflow-use-tf-slim/>


###my-tf-alexnet


* 使用tensorflow（1.2版本）自己手动搭建的Alexnet
* 用于熟练tensorflow的使用方法，比如数据喂入网络的阶段，使用了tfrecord+queue的方法降低I/0与内存的读取消耗
* 熟悉CNN各基本layer的定义细节（如fc层实际上是n个全局卷积的结果)
* 熟悉神经整个运行流程，比如最后的输出logits作用，一是用于softmax+crossentropy计算损失，二是也能用于计算精确度

