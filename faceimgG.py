#encoding: utf8

import numpy as np
import cv2 
 
# 顔検出対象の画像をロードし、白黒画像にしておく。
imgS = cv2.imread('face.jpg')
imgG = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
 
# 学習済みデータをロード
classifier = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(classifier)
# 顔検出を実行
faces = cascade.detectMultiScale(imgG, 1.3, 5)
 
# 顔検出箇所を矩形で囲む。
for (x,y,w,h) in faces:
    cv2.rectangle(imgS,(x,y),(x+w,y+h),(0,255,255),2)
# 表示
cv2.imshow('image',imgS)
cv2.waitKey(0)
 
# 表示消去
cv2.destroyAllWindows()

