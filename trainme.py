import cv2
import numpy as np
from PIL import Image
import os
## Intialization of data location amd creating a Linear binary pattern histogram based recognizer ##
path_gen = 'data' 
recognizer = cv2.face.LBPHFaceRecognizer_create()
## Haarcasde for face detection in given dataset ##
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
## Getting images and label from images in given path for training
def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(id)
        faces = detector.detectMultiScale(img_numpy)
        print(faces)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
paths= []
faces_main= []
ids_main=[]
with os.scandir(path_gen) as dirs:
    for d in dirs:
        new_path = os.path.join(d)
        paths.append(new_path)
        print(new_path)
print(paths)
## For every folder in given data path fetch the images and load them for detection and labeling ##
for path in paths:
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    for face in faces:
        faces_main.append(face)
    for i in ids:
        ids_main.append(i)
print(ids_main)
## Train the fetched faces and ids in recognizer ## 
recognizer.train(faces_main, np.array(ids_main))
print(recognizer)
# Save the model into trainer/trainer.yml #
recognizer.write('trainer/trainer.yml')
# Print the numer of faces trained and end program #
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
# :- Written By piyushS3V3N 