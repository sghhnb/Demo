# -*- coding: utf-8 -*-
import A_tfrecords
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    img_batch, label_batch = A_tfrecords.get_tfrecord(19000, isTrain=True)
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess, coord)
    data, label = sess.run([img_batch, label_batch])
    print(data[0:5])
    print(label[0:5])
    print(data.shape)
    print(label.shape)
    print("How does the training data look like?")
    nsample = 10
    for i in range(6):
        print("i=",i)
        img_batch1 = data[i*1000:i*1000+1000,:]
        label_batch1 = label[i*1000:i*1000+1000,:]
        print("img_batch1",img_batch1.shape)
        print("label_batch1", label_batch1.shape)
        randidx = np.random.randint(img_batch1.shape[0], size=nsample)
        for i in randidx:
            curr_img = np.reshape(img_batch1[i, :], (40, 40))
            curr_label = np.argmax(label_batch1[i, :])  # Label
            plt.matshow(curr_img, cmap=plt.get_cmap('gray'))

            plt.title(" " + str(i) + "th Training Data " + "Label is " + str(curr_label))
            print("" + str(i) + "th Training Data " + "Label is " + str(curr_label))
            plt.show()