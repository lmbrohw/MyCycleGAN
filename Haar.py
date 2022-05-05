import cv2
import numpy as np
import matplotlib.pyplot as plt
# 1.加载文件和图片 2.进行灰度处理 3.得到haar特征 4.检测人脸 5.标记

face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('/Users/mac/Downloads/WechatIMG27.jpg')
# cv2.imshow('img', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 1.灰色图像 2.缩放系数 3.目标大小
faces = face_xml.detectMultiScale(gray, 1.3, 5)
print('face = ',len(faces))
print(faces)

#绘制人脸，为人脸画方框
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 2)
    # roi_face = gray[y:y+h,x:x+w]
    # roi_color = img[y:y+h,x:x+w]
    # eyes = eye_xml.detectMultiScale(roi_face)
    # print('eyes = ',len(eyes))
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color, (ex,ey),(ex + ew, ey + eh), (0,255,0), 2)
# cv2.imshow('dat', img)
cv2.imwrite('img1.jpg', img)