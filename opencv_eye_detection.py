import cv2 
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
eye_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        region_gray = gray[y:y+h,x:x+w]
        region_color = img[y:y+h,x:x+w]
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        eyes = eye_cascade.detectMultiScale(region_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(region_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    cv2.imshow("FACE",img)
    if (cv2.waitKey(5) & 0xFF == ord("Q")):
        break
cap.release()
cv2.destroyAllWindows()
