#!/usr/bin/env python
#encoding: utf8
import numpy as np
import cv2

# メイン関数
image = cv2.imread('dau.jpg') # ファイル読み込み

    # HSVでの色抽出
hsvLower = np.array([0, 100, 0])    # 抽出する色の下限
hsvUpper = np.array([5, 255, 255])    # 抽出する色の上限
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # 画像をHSVに変換
hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)    # HSVからマスクを作成
result = cv2.bitwise_and(image, image, mask=hsv_mask) # 元画像とマスクを合成
cv2.imshow('reddetect.jpg', result)
