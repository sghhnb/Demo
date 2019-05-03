# -*- coding: utf-8 -*-
import cv2
import os
import shutil

if __name__ == '__main__':

    testdir = "E:\PAPER\mnist_data_jpg\mnist_test_jpg1"
    testoo = "E:\PAPER\mnist_data_jpg\mnist_test_jpg"
    traindir = "E:\PAPER\mnist_data_jpg\mnist_train_jpg1"
    trainoo = "E:\PAPER\mnist_data_jpg\mnist_train_jpg"

    testgray = "E:\PAPER\mnist_data_jpg\mnist_test_jpg1"
    traingray = "E:\PAPER\mnist_data_jpg\mnist_train_jpg1"
    image_list = os.listdir(trainoo)
    '''
    for i in range(0, len(image_list)):
        img_path = os.path.join(traindir, image_list[i])
        re_file = os.path.join(trainoo, image_list[i])
        crop_size = (40, 40)
        img = cv2.imread(img_path)
        print(img_path)
        image_new = cv2.resize(img, crop_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(re_file, image_new)
    '''
    for i in range(0, len(image_list)):
        img_path = os.path.join(trainoo, image_list[i])
        re_file = os.path.join(traingray, image_list[i])
        crop_size = (40, 40)
        img = cv2.imread(img_path)
        image_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(re_file, image_new)