import cv2
import numpy as np
import os 
import json
# names related to ids: example ==> Piyush: id=1,  etc
name_data = json.loads(open('indexdata.json').read())
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
print(cam.get(3),cam.get(4))
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically due to image orientation problem in Droidcam
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #faces = faceCascade.detectMultiScale( gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)),)
    faces = faceCascade.detectMultiScale( gray,scaleFactor = 1.2,minNeighbors = 5)
    #print(faces)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (100,100,100), 1)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 60):
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 1)
            #print(confidence)
            id = str(name_data['Family']['names'][id])
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1.5, (0,255,0), 1)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
# :- Written By piyushS3V3N
