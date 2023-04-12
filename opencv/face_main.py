import numpy as np
import cv2 as cv
import uuid
import csv

haar_cascade=cv.CascadeClassifier('haar_face.xml')
people=['wanted_terrorist']

# features=np.load('features.npy')
# labels=np.load('labels.npy')

face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cap=cv.VideoCapture(0)
while True:
    ret,frame=cap.read()
    # img=cv.imread('terrorist.jpg')
    gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces
    faces_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces_rect:
        faces_roi=gray[y:y+h, x:x+w]

        label, confidence=face_recognizer.predict(faces_roi)
        print(label)
        print(confidence)

        cv.putText(frame, str(label), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    cv.imshow('Person Recognised', frame)
    cv.waitKey(0)