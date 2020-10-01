#encoding: utf8

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

def danh_dau_moc(im):
    rects = detector.detectMultiScale(im, 1.3,5)
    x,y,w,h =rects[0]
    rect=dlib.rectangle(x,y,x+w,y+h)

    shape = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    print(shape)
    return shape
    # for p in predictor(im, rect):
    #     print(p)
    # return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def ty_so_mat(eye):
    print(eye)
# tinh khoang cach euclide giua 2 bo danh dau mat doc toa do
    # (x, y)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # tinh khoang cach euclide giua diem moc danh dau mat ngang toa do (x, y)
    C = dist.euclidean(eye[0], eye[3])
    # tinh ti le mat
    ear = (A + B) / (2.0 * C)
    # tra ve ti le mat
    return ear

NGUONG_MAT = 0.24
SO_KHUNG_HINH = 10
DEM = 0


cascade_path='haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade_path)

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
vs = VideoStream(src=0).start()
time.sleep(1.0)
print ( "--- %s seconds ---" % (time.time() - start_time))

while True:
    start_time_1 = time.time()
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for rect in rects:
        shape = danh_dau_moc(gray)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = ty_so_mat(leftEye)
        rightEAR = ty_so_mat(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
# tinh vien bao loi cho mat trai va mat phai,
# sau do hinh dung ra (ve ra) vien bao do cho moi mat
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < NGUONG_MAT:
            DEM += 1
            if DEM >= SO_KHUNG_HINH:
                    cv2.putText(frame, "NGU GAT!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        else:
            DEM = 0
        cv2.putText(frame, "TY SO MAT: {:.2f}".format(ear), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0,0), 2)
    cv2.imshow("PHAT HIEN NGU GAT", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    print ( "--- %s seconds ---" % (time.time() - start_time_1))
cv2.destroyAllWindows()
vs.stop()
