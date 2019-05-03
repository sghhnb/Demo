# -*- coding: utf-8 -*-
import cv2
import os

def draw(image_list, src_image_dir, tar_image_dir):
    if not os.path.exists(tar_image_dir):
        os.mkdir(tar_image_dir)
    for item in image_list:
        i = 0
        sub_path = item["path"]
        path_seg = sub_path.split("/")
        path = os.path.join(src_image_dir, sub_path)
        boxes = item["boxes"]
        image = cv2.imread(path)
        tar_dir = os.path.join(tar_image_dir, path_seg[0])
        if not os.path.exists(tar_dir):
            os.mkdir(tar_dir)
        for box in boxes:
            i += 1
            para = box.split(" ")
            x, y, w, h = int(para[0]), int(para[1]), int(para[2]), int(para[3])
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 1)
            str1 = "Num" + str(i) +  "_" + path_seg[1]
            tar_path = os.path.join(tar_dir, str1)
            cv2.imwrite(tar_path, image[y:(y+h), x:(x+w)])

        #tar_path = os.path.join(tar_dir, path_seg[1])
        #cv2.imwrite(tar_path, image)

def parse(label_file_path, src_img_dir, tar_img_dir):
    fr = open(label_file_path, 'r')
    image_list = []
    line = fr.readline().rstrip()
    while line:
        mdict = {}
        path = line
        mdict["path"] = path
        num = fr.readline().rstrip()
        boxes_list = []
        for n in range(int(num)):
            box = fr.readline().rstrip()
            boxes_list.append(box)
        mdict["boxes"] = boxes_list
        image_list.append(mdict)
        line = fr.readline().rstrip()
    draw(image_list, src_img_dir, tar_img_dir)

if __name__=='__main__':
    file_path = "E:\PyCharmFile\paper\WIDER_FACE\wider_face_split\wider_face_train_bbx_gt.txt"
    source_image_dir = "E:\PyCharmFile\paper\WIDER_FACE\WIDER_train\images"
    target_image_dir = "E:\PyCharmFile\paper\TargetImage"
    parse(file_path, source_image_dir, target_image_dir)