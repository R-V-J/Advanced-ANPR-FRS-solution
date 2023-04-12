import threading
import cv2 as cv
from deepface import DeepFace

cap=cv.VideoCapture(0, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

counter=0
face_match=False
reference_img=cv.imread('WhatsApp.jpg')

while True:
    ret, frame=cap.read()
    if ret:
        cv.