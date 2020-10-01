#encoding: utf8
#nhap nhung call can thiet 
from scipy.spatial import distance as dist 
from imutils.video import VideoStream 
from imutils import face_utils 
from threading import Thread 
import numpy as np 
import imutils 
import time 
import dlib 
import cv2 

start_time = time.time()
def danh_dau_moc (im): 
    rects = detector.detectMultiscale (im, 1.3, 5)
    x, y, w, h = rects [0] 
    rect = dlib.rectangle (x, y, x + w, y + h) 
    return np.matrix ([[p.x, p.y] for p in predictor(im, rect). parts()])
def ty_so_mat (eye): 
#tính khoảng cách euclide giua 2 bộ danh mục đầu chiếu dọc theo (x, y) 
    A = dist.euclidean(eye[1], eye[5]) 
    B = dist.euclidean(eye[2], eye[4])
#tính khoảng cách euclide giữa điểm mốc đánh dấu mắt ngang tọa độ (x,y) 
    C = dist.euclidean(eye[0], eye[3])
#tính tỉ lệ mắt
    ear = (A+B)/ (2.0*C)
#trả về tỉ lệ mắt
    return ear
#tỉ lệ khía cạnh mắt để biết mắt nhấp nháy và số khung liên tiếp mà mắt nằm dưới ngưỡng
NGUONG_MAT = 0.25
SO_KHUNG_HINH = 10

DEM = 0

#kho tao bo phat hien khuon mat cua opencv (dua tren haar cascade) va tao ra bo du doan danh dau moc cua khuon mat cua dlib 
#print ("[INO) 1oading Face mốc dự đoán") 
cascade_path = "haarcascade_frontalface_default.xml"
detector = cv2 .CascadeClassifier(cascade_path) 
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" 
predctor = dlib.shape_predictor(PREDICTOR_PATH) 
#lay ca chi so cua cac dau moc tren khuon mat cho mat trai va mat phai tuong ung
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS ["left_eye"] 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS ["right_eye"]
# bat dau thu luong video 
#print ("[INFO] bắt đầu luồng video") 
vs = VideoStream (src = 0).start ()  
time.sleep (1.0) 
print ("--- (time.time () – start_time) seconds ---" )
#vong lap qua cac khung hinh tu luong video 
while True: 
    start_time_1 = time.time () 
#lay khung tu luong tep video, thay doi kich thuoc va chuyen doi no sang cac kenh mau xam 
    frame = vs.read () 
    frame = imutils.resize (frame, width = 400) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
#khung phát hiện các khuon mat trong anh xam 
    rects = detector.detectMultiScale (gray, 1.3, 5)
#vong lap tren phat hien khuon mat cho direct 
    for rect in rects: 
#xac dinh diem moc tren mat doi vol vung khuon mat, sau do chuyen doi diem moc tren mat do (x, y) voi mang NumPy 
        shape = danh_dau_moc (gray) 
#trich xuat toa do mat trai va phai, sau do su dung toa do de tinh ti le mat cho ca 2 mat 
        leftEye = shape[lStart:lEnd] 
        rightEye = shape[rStart:rEnd]
        leftEAR = ty_so_mat(leftEye) 
        rightEAR = ty_so_mat(rightEye)
#tinh ti le mat trung binh cho ca 2 mat 
        ear = (leftEAR + rightEAR) / 2.0 
#tinh vien bao loi cho mat trai va mat phai, sau do hinh dung ra (ve ra) vien bao cho moi mat
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye) 
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#kiem tra ti le mat co nam duoi nguong nhay mat hay neu co tang bo dem khung nhap nhay 
        if ear < NGUONG_MAT: 
            DEM += 1  
#neu nham mat du so luong khung da dat thi bao dong 
            if DEM >= SO_KHUNG_HINH: 
                cv2.putText (frame, "NGU GAT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
# mat khac neu ty le mat khong duoi nguong nhay mat 
        else:  
            DEM = 0 
# ve thong so ty le mat da tinh tren khung de giup viec kiem tra sua loi va thiet 1ap lal dung nguong ty le mat va bo dem khung 
        cv2.putText (frame, "TY SO MAT: {: .2f}".format(ear), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 2)
#show khung hinh 
    cv2.imshow('PHAT HIEN NGU GAT', frame) 
    key = cv2.waitKey (1) & 0xFF 
#nhan'q 'de thoat khoi vong lap va xoa ngo ra pi 
    if key == ord ("q"): 
        break 
    print ("--- (time.time() - start_time_1) seconds ---" ) 
#ngung thu video va dong tat ca cua so 
cv2.destroyAllWindows () 
vs.stop ()

