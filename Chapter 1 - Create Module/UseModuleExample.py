import cv2
import time
import mediapipe as mp
from HandTrackingModule import handDetector

#for fps
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

detector = handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img) # Try findPosition(img ,draw = False) for different
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time() #Get current time
    fps = 1/(cTime - pTime) #calculate frames per second
    pTime = cTime 

    cv2.putText(img , str(int(fps)) , (10 , 70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255 , 0 , 255) , 3) # Add current fps to image

    cv2.imshow("Image" , img)
    cv2.waitKey(1)