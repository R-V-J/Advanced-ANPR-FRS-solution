import cv2 as cv
import os
import numpy as np
import face_detection

img=cv.imread('terrorist.jpg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
haar_cascade=cv.CascadeClassifier('haar_face.xml')

faces_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
print(len(faces_rect))
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

cv.imshow('terrorist', img)
cv.waitKey(0)