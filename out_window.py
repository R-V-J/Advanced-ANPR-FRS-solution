# Modified by Augmented Startups & Geeky Bee
# October 2020
# Facial Recognition Attendence GUI
# Full Course - https://augmentedstartups.info/yolov4release
# *-
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, QDate, Qt
from PyQt5.QtWidgets import QDialog,QMessageBox
import cv2
import face_recognition
import numpy as np
import datetime
import os
import csv

class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("C:\\Users\\Himanshu Singh\\OneDrive\\Desktop\\CODES\\GUI\\Base_folder\\Face-Recogntion-PyQt\\Face_Detection_PyQt_Final\\outputwindow.ui", self)

        #Update time
        now = QDate.currentDate()
        current_date = now.toString('ddd dd MMMM yyyy')
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.date_label.setText(current_date)
        self.time_label.setText(current_time)

        self.image = None

    @pyqtSlot()
    def startVideo(self, camera_name):
        """
        :param camera_name: link of camera or usb camera
        :return:
        """
        # if len(camera_name) == 1:
        # 	self.capture = cv2.VideoCapture(int(camera_name))
        # else:
        # 	self.capture = cv2.VideoCapture(camera_name)
        # self.timer = QTimer(self)  # Create Timer
        # path = 'ImagesAttendance'
        # if not os.path.exists(path):
        #     os.mkdir(path)

        haar_cascade=cv2.CascadeClassifier('C:\\Users\\Himanshu Singh\\OneDrive\\Desktop\\CODES\\GUI\\Base_folder\\Face-Recogntion-PyQt\\Face_Detection_PyQt_Final\\haar_face.xml')
        people=['wanted_terrorist']

        # features=np.load('features.npy')
        # labels=np.load('labels.npy')

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('C:\\Users\\Himanshu Singh\\OneDrive\\Desktop\\CODES\\GUI\\Base_folder\\Face-Recogntion-PyQt\\Face_Detection_PyQt_Final\\face_trained.yml')

        cap=cv2.VideoCapture(0)
        while True:
            ret,frame=cap.read()
            # img=cv.imread('terrorist.jpg')
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces
            faces_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in faces_rect:
                faces_roi=gray[y:y+h, x:x+w]

                label, confidence=face_recognizer.predict(faces_roi)
                print(label)
                print(confidence)
                self.acc_label.setText(str(confidence))
                self.name_label.setText(str(label))

                cv2.putText(frame, str(label), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

            cv2.imshow('Person Recognised', frame)
            cv2.waitKey(0)

    def showdialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("This is a message box")
        msg.setInformativeText("This is additional information")
        msg.setWindowTitle("MessageBox demo")
        msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

    def update_frame(self):
        ret, self.image = self.cap.read()
        self.displayImage(self.image, self.encode_list, self.class_names, 1)

    def displayImage(self, image, encode_list, class_names, window=1):
        """
        :param image: frame from camera
        :param encode_list: known face encoding list
        :param class_names: known face names
        :param window: number of window
        :return:
        """
        image = cv2.resize(image, (640, 480))
        try:
            image = self.face_rec_(image, encode_list, class_names)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.img_feed.setPixmap(QPixmap.fromImage(outImage))
            self.img_feed.setScaledContents(True)
