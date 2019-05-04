#coding=utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt

print('load object cascade classifier')
faceCascode = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt2.xml')
faceCascode1 = cv2.CascadeClassifier('./xml/face.xml')
faceCascode2 = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt.xml')
faceCascode3 = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt_tree.xml')
print('load image')
imgfile = 'images/uu.jpg'
img = cv2.imread(imgfile)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

faces = faceCascode.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20),
                                         maxSize=(1024, 683))
faces1 = faceCascode1.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20),
                                           maxSize=(1024, 683))
faces2 = faceCascode2.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20),
                                           maxSize=(1024, 683))
faces3 = faceCascode3.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(20, 20),
                                           maxSize=(1024, 683))

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
    faceGray = imgGray[y:y+h,x:x+w]
    faceColor = img[y:y+h,x:x+w]

for (x, y, w, h) in faces1:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    faceGray = imgGray[y:y + h, x:x + w]
    faceColor = img[y:y + h, x:x + w]

plt.subplot(1, 1, 1), plt.imshow(img), plt.title('Face Image'), plt.xticks([]), plt.yticks([])
plt.show()
