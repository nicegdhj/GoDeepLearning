# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 20:52:04 2016

@author: wu mengying & chenli
"""

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras.optimizers import SGD
# encode and decode some digits
# note that we take them from the *test* set
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

# this is the size of our encoded representations
encoding_dim = 504  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
layer_encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input

layer_classified = Dense(10, activation='softmax')(layer_encoded)

layer_decoded = Dense(784, activation='sigmoid')(layer_classified)

# this model maps an input to its reconstruction
autoencoder_cls = Model(input=input_img, output=[layer_decoded,layer_classified])

#classifier = Model(input=input_img, output=layer_classified)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=layer_encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(10,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder_cls.layers[-1]
#classifier_layer = classifier1.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
#classifier =  Model(input=encoded_input, output=classifier_layer(encoded_input))
sgd = SGD(lr=0.01, momentum=1e-6, decay=0.9, nesterov=True)

autoencoder_cls.compile(optimizer='adam', loss=['binary_crossentropy','mse'])
#classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



(x_train, train_label),(x_test, test_label) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

label1=np.zeros(shape=(60000,10))
label2=np.zeros(shape=(10000,10))



for i in range(len(train_label)):
    label1[i][train_label[i]]=1

for j in range(len(test_label)):
    label2[j][test_label[j]]=1

x_valid = x_train[50000:,]
x_train = x_train[0:50000,]
label_valid = label1[50000:,]
label1 = label1[0:50000,]

autoencoder_cls.fit(x_train, [x_train,label1],
                verbose=2,
                nb_epoch=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_valid, [x_valid,label_valid]))
                
#encoded_imgs_train = encoder.predict(x_train)
#decoded_imgs_train = decoder.predict(encoded_imgs_train)
#
#encoded_imgs_test = encoder.predict(x_test)
#decoded_imgs_test = decoder.predict(encoded_imgs_test)
#
##
#classifier.fit(encoded_imgs_train, label1,
#                nb_epoch=1,
#                batch_size=256,
#                shuffle=True,
#                validation_data=(encoded_imgs_test, label2))
#                
#classified_imgs = classifier.predict(encoded_imgs_test)

[decoded_imgs_test,classified_imgs] = autoencoder_cls.predict(x_test)

labels = np.argmax(classified_imgs, axis = 1)
accuracy = np.sum(labels == test_label)/float(10000)
print accuracy 



n = 20  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#
##print classified_imgs