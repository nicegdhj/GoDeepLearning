from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

"""
MNIST数据集的图片大小均为28*28，进入网络时转换成1*784， 经过encoded层之后输出1*32的
压缩图片，在经过decoded层恢复为原图片。
"""

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
                   # 压缩率为 784/32 = 24.5

# this is our input placeholder 
# 输入图片转换为1*784.
input_img = Input(shape=(784,)) 
# "encoded" is the encoded representation of the input
# 图片经过encoded层， 激活函数为relu。
encoded = Dense(encoding_dim, activation='relu')(input_img) 
# "decoded" is the lossy reconstruction of the input
# 压缩图片再经过decoded层重新恢复为原图。激活函数选用sigmoid。
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
#建立autoencoder模型。
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
#建立单独的编码器模型。
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
#以下为建立一个单独的解码器的步骤。
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
# 获取自动编码器模型的最后一层。
decoder_layer = autoencoder.layers[-1]
# create the decoder model
#建立解码器模型
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

#对autoencoder进行编译
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#mnist数据预处理
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

#进行训练
autoencoder.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
                
# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)

#测试
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
