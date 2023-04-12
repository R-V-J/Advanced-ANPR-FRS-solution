import os
import cv2 as cv
import numpy as np

people=['wanted_terrorist', 'barack_obama', 'mrunal', 'baba']
DIR=R'D:\training_photos'
haar_cascade=cv.CascadeClassifier('haar_face.xml')

features=[]
labels=[]

def create_train():
    for person in people:
        path=os.path.join(DIR, person)
        label=people.index(person)

        for img in os.listdir(path):
            img_path=os.path.join(path, img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=12)

            for (x, y, w, h) in faces_rect:
                faces_roi=gray[y:y+h+1, x:x+w+1]
                features.append(faces_roi)
                labels.append(label)


create_train()
print('Length of the features list', len(features))
print('Length of the labels list', len(labels))
print('training done')

features=np.array(features, dtype='object')
labels=np.array(labels)
face_recognizer=cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the features and labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)