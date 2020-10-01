#encoding: utf8

import cv2, sys, numpy, os 
webcam = cv2.VideoCapture(0)  
while(1):
  
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
# 学習済みデータをロード
    classifier = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(classifier)
# 顔検出を実行
    faces = cascade.detectMultiScale(gray, 1.3, 4) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 

    cv2.imshow('OpenCV', im) 
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
