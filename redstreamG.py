#encoding: utf8
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    ret, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower = np.array([0, 100, 0])
    upper = np.array([5, 255, 255])

    lower2 = np.array([160, 100, 100])
    upper2 = np.array([180, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower, upper)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Bitwise-AND mask and original image
    red = cv2.bitwise_and(frame,frame, mask= mask)

    # グレースケール変換
    gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

    # 2値化
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # ラベリング処理
    label = cv2.connectedComponentsWithStats(gray)

    # ブロブ情報を項目別に抽出
    n = label[0] - 1   #　ブロブの数
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    # ブロブ面積最大のインデックス
    try:
        max_index = np.argmax(data[:,4])
    # 面積最大ブロブの各種情報を表示
        center = center[max_index]
        x1 = data[:,0][max_index]
        y1 = data[:,1][max_index]
        x2 = x1 + data[:,2][max_index]  # x1 + 幅
        y2 = y1 + data[:,3][max_index]  # y2 + 高さ
        a = data[:,4][max_index]        # 面積
        result1 = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255))
        result2 = cv2.rectangle(red, (x1, y1), (x2, y2), (0, 0, 255))
        cv2.imshow('frame',result1)
        cv2.imshow('red',result2)
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break
    except:
        print("khong co mau do")

cv2.destroyAllWindows()
