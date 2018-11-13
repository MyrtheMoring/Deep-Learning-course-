"""
SVHN: The Street View House Numbers
Author: Myrthe Moring
Date: 22/10/2018

1. Loading data (train and test data)
2. Preprocessing data
    2.1 Convert to grayscale
    2.2 Normalize data
3. CNN model
4. Plot results
    4.1 Accuracy and loss model
    4.2 Confusion matrix
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import keras
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from skimage import io, color
import scipy.io as sio
from scipy.io import loadmat
import seaborn as sns
import os
import time
from datetime import timedelta
import h5py


def get_data(file):
    """ Return tuple with:
    - x = 4-D matrix containing the images x (images)
    - y = 1-D vector of class labels (0-10) """
    data = sio.loadmat(file)
    x = data['X']
    y = data['y']
    y[y == 1] = 0
    return (x,y)

(x_train, y_train), (x_test, y_test) = get_data('train_32x32.mat'),get_data('test_32x32.mat')
x_train, x_test = x_train.transpose(3,0,1,2), x_test.transpose(3,0,1,2)
y_train[y_train == 10] = 0
y_test[y_test == 10] = 0

"Get the validation data"
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train[:,0], test_size=0.13, random_state=7)
y_val[y_val == 10] = 0

# The number of images
nr_images_train = x_train.shape[0]
nr_images_val = x_val.shape[0]
nr_images_test = x_test.shape[0]
print('Total number of images', nr_images_train + nr_images_test + nr_images_val)

# The sizes of dataset
print('Size of train examples', nr_images_train, "& size of images", x_train.shape[1:3])
print('Size of validation examples', nr_images_val, "& size of images", x_val.shape[1:3])
print('Size of test examples', nr_images_test, "& size of images", x_test.shape[1:3])

""" Make the class distribution plots: """
# plots, (train_ax, test_ax, val_ax) = plt.subplots(1,3)
# plots.suptitle('Data distribution', fontsize=18, y=1.05)
# plots.set_size_inches(12, 8)
#
# train_ax.hist(y_train, color='r', bins=10)
# train_ax.set_title("Training set: class distribution", y=1)
# train_ax.set_xlim(1, 10)
# train_ax.set_xlabel("Image class")
# train_ax.set_ylabel("Number of images")
#
# val_ax.hist(y_val, color='g', bins=10)
# val_ax.set_title("Validation set: class distribution", y=1)
# val_ax.set_xlim(1, 10)
# val_ax.set_xlabel("Image class")
# val_ax.set_ylabel("Number of images")
#
# test_ax.hist(y_test, color='b', bins=10)
# test_ax.set_title("Test set: class distribution", y=1)
# test_ax.set_xlim(1, 10)
# test_ax.set_xlabel("Image class")
# test_ax.set_ylabel("Number of images")
#
# plots.tight_layout()
# plots.savefig('classdistribution.png')

"""Reshape the labeled data and One Hot Label Encoding. """
y_train, y_val, y_test = y_train.reshape(nr_images_train), y_val.reshape(nr_images_val), y_test.reshape(nr_images_test)
y_train, y_val, y_test = np_utils.to_categorical(y_train, 10), np_utils.to_categorical(y_val, 10), np_utils.to_categorical(y_test, 10)


# def plot_images(x_train, y_train, classes, title="Images", save=False):
#     """ Plot the images with the labels """
#     figure, ax = plt.subplots(1, classes)
#     figure.suptitle(title, y=0.7)
#     for i, j in enumerate(ax.flat):
#         if x_train[i].shape == x_train[-1].shape:
#             if x_train.shape[3] == 1:
#                 j.imshow(x_train[i,:,:,0], cmap='gray')
#             else:
#                 j.imshow(x_train[i,:,:,:])
#             lbl = argmax(y_train[i])
#             j.set_title(lbl)
#             j.set_xticks([])
#             j.set_yticks([])
#         else:
#             print("Not valid images")
#     if save:
#         figure.savefig(title+".png")
#
# plot_images(x_train, y_train, 12, "Train images")
# plot_images(x_val, y_val, 15, "Validation images")
# plot_images(x_test, y_test, 15, "Test images")

def rgb2gray(data):
    """ Grey = 0.299 R + 0.587 G + 0.114 B  """
    grey = np.expand_dims(np.dot(data, [0.2990, 0.5870, 0.1140]), axis=3)
    return grey

train_gray, val_gray, test_gray = rgb2gray(x_train).astype(np.float32), rgb2gray(x_val).astype(np.float32), rgb2gray(x_test).astype(np.float32)

# plot_images(train_gray, y_train, 15, "Grey train images")
# plot_images(val_gray, y_val, 15, "Grey validation images")
# plot_images(test_gray, y_test, 15, "Grey test images")

def normalize_images(img):
    """  Normalize data: to change the range of pixel intensity valuesself.
    The linear normalization of a grayscale digital image is performed according to the formula (Wikipedia):
    norm = (image - mean)/std """

    mean,std = np.mean(img, axis=0), np.std(img, axis=0)
    return ((img - mean) / std)

train_norm = normalize_images(train_gray)
val_norm = normalize_images(val_gray)
test_norm = normalize_images(test_gray)

# plot_images(train_norm, y_train, 15, "Normalized train images")
# plot_images(val_norm, y_val, 15, "Normalized validation images")
# plot_images(test_norm, y_test, 15, "Normalized test images")

x_train = train_norm
x_val = val_norm
x_test = test_norm

def CNN_model(ftrs, d, c, r):
    """Model function for CNN."""

#   Input Layer
    l1 = tf.reshape(ftrs, [-1, d, d, 1])

#   First 2D convolutional layer
    c1 = tf.layers.conv2d(
        inputs=l1,
        filters=d,
        kernel_size=[8, 8],
        padding="same",
        activation=tf.nn.relu)

#   First max pooling layer (2,2)
    p1 = tf.layers.max_pooling2d(inputs=c1, pool_size=[2, 2], strides=2)

#   Second 2D convolution layer
    c2 = tf.layers.conv2d(
          inputs=p1,
          filters=2*d,
          kernel_size=[8, 8],
          padding="same",
          activation=tf.nn.relu)

#   Second max pooling layer (2,2)
    p2 = tf.layers.max_pooling2d(inputs=c2, pool_size=[2, 2], strides=2)

#   Flatten Layer
    flat = tf.reshape(p2, [-1, 128*d])

#   Dense Layer
    dense = tf.layers.dense(inputs=flat, units=256, activation=tf.nn.relu)

#   Dropout layer
    fc = tf.layers.dropout(inputs=dense, rate=r, seed=None)

#   Return the fully connected Layer
    output = tf.layers.dense(inputs=fc, units=c)

    return output

# Get batch dynamically
def dynamic_batch(x, y, b):
    for i in np.arange(0, y.shape[0], b):
        end = min(x.shape[0], i + b)
        yield(x[i:end],y[i:end])

""" Model: using a tenserflow CNN architecture.
    Parameters:
    - train, test, val: image and label data
    - batch: the size of the batch
    - epochs_max: the maximum size of the epochs
    - rate: the discard rate
    - m: step size
    """

def run_CNN_model(train, test, val, batch, epochs_max, rate, m):

    (xt, yt) = train
    (xts, yts) = test
    (xv, yv) = val

#   1: Tenserflow placeholder variables
    (d1, d2) = xt.shape[1:3]
    classes = yt.shape[1]
    img = tf.placeholder(tf.float32, shape = [None, d1, d2, 1])
    label = tf.placeholder(tf.float32, shape = [None, classes])
    discard_rate = tf.placeholder(tf.float32)

#   2: Get prediction labels
    pred = CNN_model(img, d1, classes, discard_rate)
    pred_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))

#   3: Tenserflow loss, optimization and accuracy varibables
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=label, logits=pred))
    opt = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))

#   4: Start Tenserflow session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

#   5: Store values
    values_store = tf.train.Saver()
    path_store = os.path.join('values_store/', 'SVHN')
    values_store.save(sess=session, save_path=path_store)

#   6: Initialize lists and time
    train_loss, val_loss = [], []
    train_accuracy, val_accuracy = [], []
    t = time.time()

#   7: Set up each epoch
    print ('Training the CNN model: \n')
    for e in range(epochs_max):
        s = 0
        print ('Epoch number: ', e+1, ': out of: ', epochs_max)

        # Training each of the epochs
        for (epoch_img , epoch_label) in dynamic_batch(xt, yt, batch):
            train_opt, train_acc, train_l = session.run([opt, accuracy, loss],
                                                     feed_dict={img: epoch_img, label: epoch_label, discard_rate: rate})


            if(s%m == 0):
                acc = 0.0
                for (epoch_img , epoch_label) in dynamic_batch(xv, yv, batch):
                    correct, val_l = session.run([pred_correct, loss], feed_dict={img: epoch_img, label: epoch_label, discard_rate: 0.0})
                    acc += np.sum(correct[correct == True])
                val_acc = acc*100.0/yv.shape[0]

                val_accuracy.append(val_acc)
                val_loss.append(val_l)
                train_loss.append(train_l)
                train_accuracy.append(train_acc*100.0)

                print ("Step:", s+1, "\n")
                print('Train accuracy :' , train_acc*100.0, '%\n')
                print('Validation accuracy :', val_acc, '%\n')
#                 yield train_acc, val_acc, train_loss, val_loss
            s += 1

    # 8: Final test accuracy and time usage
    acc = 0.0
    test_pred = []
    for (epoch_img , epoch_label) in dynamic_batch(xts, yts, batch):
        correct = session.run([pred_correct], feed_dict={img: epoch_img, label: epoch_label, discard_rate: 0.0})
        test_pred.append((np.asarray(correct, dtype=int)).T)
        sum_correct = np.sum(correct[correct == True])
        acc += sum_correct
    test_acc = acc*100.0/yts.shape[0]
    print('Test accuracy :' , test_acc, '%')

    # Calculate net time
    diff = str(timedelta(seconds=int(round(time.time() - t))))

    print("Time usage: " + diff + "\n")

    return [(train_loss, val_loss), (train_accuracy, val_accuracy)]


""" Run the CNN model with the given parameters:
    - batch size = 64
    - epoch size = 2
    - dropout rate = 0.2
    - step size = 40 """
output = run_CNN_model((x_train, y_train),(x_test, y_test), (x_val, y_val), 64, 10, 0.2, 40)
[(train_loss, val_loss), (train_accuracy, val_accuracy)]  = output

# with open("train_acc_1.txt", "w") as f:
#     for s in train_accuracy:
#         f.write(str(s) +"\n")
# with open("train_loss_1.txt", "w") as f:
#     for s in train_loss:
#         f.write(str(s) +"\n")
# with open("val_acc_1.txt", "w") as f:
#     for s in val_accuracy:
#         f.write(str(s) +"\n")
# with open("val_loss_1.txt", "w") as f:
#     for s in val_loss:
#         f.write(str(s) +"\n")
