import cv2
import os
from add_people import add_user
# Start the Video Capture at Device id 0 (webcam)
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
# For each person, enter one numeric face id and a folder is created for that face id to store data (image) in it
face_id = input('\n enter user id end press <return> ==>  ')
## if user doesn't exist take its input name and add it 
try:
    #count = 0
    os.mkdir("data/"+face_id)
    name = input("Enter the name of Person : ")
    section = input("Enter the relation type : ")
    add_user(name, section,face_id)
except:
    #count += 50
    print("Directory Exist")

print("\n [INFO] Initializing face capture. Look at1the camera and wait ...")
# Initialize individual sampling face count

count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 1)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("data/"+face_id+"/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(roi_gray,1.2,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(255,255,0),1)
        data_percent = str(count/2)+"%"
        cv2.putText(img, str(data_percent), (x+5,y-5),  cv2.FONT_HERSHEY_SIMPLEX, 1,(100,124,200), 2)
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 200: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
os.system("python trainme.py")
# :- Written By piyushS3V3N