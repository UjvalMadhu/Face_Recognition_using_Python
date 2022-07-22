# Program to read a face image data set and use it to train the face recognition model

import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
id_list =[]
face_list =[]

for i in os.listdir('Captured_faces'):
    id = i.split('_')[0]
    id = int(id[1:])
    filepath = './Captured_Faces/'+ i
    facesample = cv2.imread(filepath,0) # 0 is flag which specifies to read the image as grayscale
    cv2.imshow("Sample",facesample)
    id_list.append(id)
    face_list.append(facesample)
    end = cv2.waitKey(1)
    if end == 113 or end == 81:
        cv2.destroyAllWindows()
        break
print(id_list)

recognizer.train(face_list,np.array(id_list))
print("Training on Data...")
recognizer.write("Face_recongnizer_model.xml")
