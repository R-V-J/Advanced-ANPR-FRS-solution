import cv2 as cv
import numpy as np
import imutils
import easyocr

img=cv.imread('numberplate.jpeg')
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
bifilter=cv.bilateralFilter(gray, 11, 17, 17)
edges=cv.Canny(bifilter, 30, 200)
keypoints=cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours=imutils.grab_contours(keypoints)
contours=sorted(contours, key=cv.contourArea, reverse=True)[:10]#sorting in descending order
# looping through each contour to know which one represents a quadrilateral

location=None
for contour in contours:
    approx=cv.approxPolyDP(contour, 10, True)
    if len(approx)==4:
        location=approx
        break
# print(location)

mask=np.zeros(gray.shape, np.uint8)
new_img=cv.drawContours(mask, [location], 0,255, -1)
new_img=cv.bitwise_and(img, img, mask=mask)

# cropping the image to show only the number plate
(x, y)=np.where(mask==255)
(x1, y1)=(np.min(x), np.min(y))
(x2, y2)=(np.max(x), np.max(y))
cropped_img=gray[x1:x2+1, y1:y2+1]

# reading the gray image using easyocr
reader=easyocr.Reader(['en'])
result=reader.readtext(cropped_img)
print(result)

cv.imshow('window', cropped_img)
cv.waitKey(0)