import numpy as np
import cv2

im = cv2.imread('tomato.jpg')
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower = np.array([0, 100, 0])
upper = np.array([5, 255, 255])

lower2 = np.array([160, 100, 100])
upper2 = np.array([180, 255, 255])

# Threshold the HSV image to get only red colors
mask1 = cv2.inRange(hsv, lower, upper)
mask2 = cv2.inRange(hsv, lower2, upper2)
mask = cv2.bitwise_or(mask1, mask2)

# Bitwise-AND mask and original image
red = cv2.bitwise_and(im,im, mask= mask)

gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
ret, bin_img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 小さい輪郭は誤検出として削除する
contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
cv2.drawContours(im, contours, -1, (0, 255, 0), 2) # vẽ lại ảnh contour vào ảnh gốc
for i in range(0,len(contours)):
    print(cv2.contourArea(contours[i]))
# show ảnh lên
cv2.imshow("im", im)
cv2.imshow("red", red)


cv2.waitKey(0)

cv2.destroyAllWindows()