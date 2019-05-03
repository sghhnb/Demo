# -*- coding: utf-8 -*-
import cv2
import os
import shutil

if __name__=='__main__':
    src_dir = "E:\下载\人脸数据集\WIDER_FACE\TargetImage"
    image_dir = "E:\PAPER\src_images"
    image_list = os.listdir(image_dir)
    Re_SelectFace = "E:\PAPER\Re_images"
    #复制剪切的人脸图片到src_images
    for dirpath, dirnames, filenames in os.walk(src_dir):
        for dirname in dirnames:
            src_path = os.path.join(src_dir, dirname)
            src_list = os.listdir(src_path)
            for i in range(0, len(src_list)):
                src_file = os.path.join(src_path, src_list[i])
                shutil.copy(src_file, image_dir)
    #删除小于1.2KB的人脸图片
    for i in range(0,len(image_list)):
        image_path = os.path.join(image_dir,image_list[i])
        if os.path.getsize(image_path) <= 1200:
            os.remove(image_path)
    
    image_list = os.listdir(image_dir)
    #将人脸图片全部变为40X40大小，用作Adaboost的正样本
    for i in range(0,len(image_list)):
        img_path = os.path.join(image_dir,image_list[i])
        re_file = os.path.join(Re_SelectFace, image_list[i])
        crop_size = (40,40)
        img = cv2.imread(img_path)
        print(img_path)
        image_new = cv2.resize(img, crop_size, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(re_file, image_new)
