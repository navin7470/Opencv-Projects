import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)

ret,frame1 = cap.read()
ret,frame2 = cap.read()
while cap.isOpened():
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(gray,(5,5),0)
    _ , thr = cv2.threshold(gblur,30,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thr,None,3)
    contours , _ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(frame1,contours,-1,(0,0,255),2)  || To draw contours
    #TO draw a ractangle
    for contour in contours:
        (x1,y1,h,w) = cv2.boundingRect(contour)  #TO take the co-ordinates
        if(cv2.contourArea(contour) > 1000):
            cv2.rectangle(frame1,(x1,y1),(x1+w,y1+h),(0,0,255),2)
    cv2.imshow("FEED",frame1)
    frame1 = frame2
    _,frame2 = cap.read()
    k = cv2.waitKey(40) & 0xFF
    if(k==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
