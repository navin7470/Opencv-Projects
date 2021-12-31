import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _,img = cap.read()
    #img = cv2.imread("navin.png")
    #img = cv2.resize(img,(500,550))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,10)
    for (x,y,w,h) in faces:
         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow("Image_Detection",img)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()