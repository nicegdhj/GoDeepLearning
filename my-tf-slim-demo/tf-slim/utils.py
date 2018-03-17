#coding=utf-8

import numpy as np

from matplotlib import pyplot as plt
from skimage.transform import resize
import cv2

# returns the top1 string
def print_prob(prob, file_path, top_k):

    synset = [l.strip() for l in open(file_path).readlines()]

    pred = np.argsort(list(prob))

    # Get top1 label
    top1 = synset[pred[-1]]
    print('Top1:')
    print ('Probability %0.3f => [%s]' % (prob[pred[-1]], top1))

    # Get top5 label
    print("Top%s: " % str(top_k))
    index_k = pred[-1: -(top_k+1): -1]
    for i in range(top_k):
        # 打印top5的预测类别和相应的概率值。
        print('Probability %0.3f => [%s]' % (prob[index_k[i]], synset[index_k[i]]))



def visualize(image, conv_output, conv_grad, gb_viz):

    output = conv_output           # [7,7,512]
    grads_val = conv_grad          # [7,7,512]
    gb_viz = np.dstack((
            gb_viz[:, :, 2],
            gb_viz[:, :, 1],
            gb_viz[:, :, 0],
        ))

    weights = np.mean(grads_val, axis = (0, 1)) 			# [512]
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [7,7]


    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = resize(cam, (224,224))
    # print(cam)

    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    # img = tf.cast(image, tf.float32)
    image = image.eval()
    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)

    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    # cam = np.float32(cam) + np.float32(img)
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)


    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False)
    imgplot = plt.imshow(img)
    ax.set_title('Input Image')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(cam_heatmap)
    ax.set_title('Grad-CAM')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(gb_viz)
    ax.set_title('guided backpropagation')

    #guided backpropagation  Deconv


    gd_gb = np.dstack((
            gb_viz[:, :, 0] * cam,
            gb_viz[:, :, 1] * cam,
            gb_viz[:, :, 2] * cam,
        ))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(gd_gb)
    ax.set_title('guided Grad-CAM')

    plt.show()


# test_image = np.random.random((224, 224, 3))*10
# test_gb = np.random.random((224, 224, 3))*10
# test_mart = np.random.random((7, 7, 512))*10
# test_gra =  np.random.random((7, 7, 512))*10
# visualize(test_image, test_mart, test_gra, test_gb)

def my_visualize(image, conv_output, conv_grad, gb_viz):
    output = conv_output  # [7,7,512]
    grads_val = conv_grad  # [7,7,512]
    gb_viz = np.dstack((
        gb_viz[:, :, 2],
        gb_viz[:, :, 1],
        gb_viz[:, :, 0],
    ))

    weights = np.mean(grads_val, axis=(0, 1))  # [512]
    cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (224, 224))
    # print(cam)

    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    # img = tf.cast(image, tf.float32)
    image = image.eval()
    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # cam = np.float32(cam) + np.float32(img)
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)


    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False)
    imgplot = plt.imshow(img)
    plt.axis('off')
    # fig.savefig('./paper_image/Input Image2', bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(cam_heatmap)
    plt.axis('off')
    # fig.savefig('./paper_image/Grad-CAM2', bbox_inches='tight')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(gb_viz)
    plt.axis('off')
    fig.savefig('./paper_image/Deconv2', bbox_inches='tight')

    # guided backpropagation  Deconv


    gd_gb = np.dstack((
        gb_viz[:, :, 0] * cam,
        gb_viz[:, :, 1] * cam,
        gb_viz[:, :, 2] * cam,
    ))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = plt.imshow(gd_gb)
    plt.axis('off')
    # fig.savefig('./paper_image/guided Grad-CAM2', bbox_inches='tight')


    plt.show()
