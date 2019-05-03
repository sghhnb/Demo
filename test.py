# -*- coding: utf-8 -*-
import cv2
import A_tfrecords
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    img_batch,label_batch = A_tfrecords.get_tfrecord(6000,isTrain=False)
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    data, label = sess.run([img_batch, label_batch])
print ("How does the training data look like?")
nsample = 5
randidx = np.random.randint(data.shape[0], size=nsample)
for i in randidx:
    curr_img = np.reshape(data[i, :], (40,40))
    curr_label = np.argmax(label[i, :] ) # Label
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))

    plt.title(" " + str(i) + "th Training Data " + "Label is " + str(curr_label))
    print ("" + str(i) + "th Training Data " + "Label is " + str(curr_label))
    plt.show()