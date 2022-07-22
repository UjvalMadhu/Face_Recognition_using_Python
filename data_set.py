# Program to capture different angles of a face from a camera,...
# ... these images can then be used to create a data set of the face...
# ... to later use it for appications like face recognition

import cv2
import time
import os

def face_Capture(faces,frame,images,id):
    for x,y,w,h in faces:
        capture = frame[y:y+h,x:x+w]
        filename = id + "_"+ str(images)+".jpg"
        cv2.imwrite('C:/Users/asus/Desktop/Srishti Robotics/6. Face Recognition/Face_Recognition_using_Python/Captured_faces/'+ filename,capture)



cam = cv2.VideoCapture(0)

# Cascade classifier is a predefined training model and the input is the training data...
# ...that enables it to detect faces.  

face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
id = input("Please enter your id: ")
images = 0   # variable to count the number of images captured

while(True):
    
    success, frame = cam.read()
    if success:
        # The camera feed is conerted from BGR to grey scale for detection purpose
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #detectMultiScale is a fuction that is used to create a dataset as an array...
        # ... of the position of faces detected from grey  
        faces = face_model.detectMultiScale(grey,minNeighbors = 5, minSize =(150,150), maxSize = (400,400) )
        
        # 'if' function to ask person to stay within camera range
        if len(faces) == 0:
            cv2.putText(frame, "Look straight and Stay within 0.3m to 1.5m from the camera",(20,30),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2)
        
        elif len(faces) > 0 and len(faces) == 1:
            cv2.putText(frame, "Try and make different Expressions and angles",(20,30),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,0),2)
            time.sleep(1)
            face_Capture(faces,frame,images,id)
            cv2.putText(frame, f"Images Captured:{images}",(20,70),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
            images+=1
            
        elif len(faces) > 1:
            cv2.putText(frame, "More than one person in frame.",(20,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
        
        # stop after capturing 30 images
        if images == 30:
            cv2.putText(frame, "Captured 30 images.",(20,30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)
            time.sleep(3)
            cv2.imshow("Face Detection",frame)
            break
        else:
            cv2.imshow("Face Detection",frame)
            
        
    end = cv2.waitKey(1)
    if end == 113 or end == 81:
        cv2.destroyAllWindows()
        break