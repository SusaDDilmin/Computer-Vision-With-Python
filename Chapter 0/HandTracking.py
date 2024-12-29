import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Hand Tracking - Using MediaPipe (like starting point)
mpHands = mp.solutions.hands
hands = mpHands.Hands() # detection of hands
mpDraw = mp.solutions.drawing_utils # drawing landmarks

#for fps
pTime = 0
cTime = 0

while True:
    success , img = cap.read()
    #img is in BGR - Convert it to RGB
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) # process the image - detection of hands
    print(results.multi_hand_landmarks) #check if hands are detected

    if results.multi_hand_landmarks: #Check whether hands are detected
        for handLms in results.multi_hand_landmarks: #draw landmarks on each hand detected
            for id , lm in enumerate(handLms.landmark): # enumerate to get id of each landmark
                print(id , lm) # print id of each landmark
                h , w , c = img.shape # get height , width , channels of image - using x , y coordinates
                cx , cy = int(lm.x * w) , int(lm.y * h) # get x , y coordinates of each landmark
                print(id , cx , cy)
                if id == 0: # Select 0th landmark
                    cv2.circle(img , (cx , cy) , 15 , (255 , 0 , 255) , cv2.FILLED) # draw circle on 0th landmark

            mpDraw.draw_landmarks(img , handLms , mpHands.HAND_CONNECTIONS) # draw connections & landmarks

    

    cTime = time.time() #Get current time
    fps = 1/(cTime - pTime) #calculate frames per second
    pTime = cTime 

    cv2.putText(img , str(int(fps)) , (10 , 70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255 , 0 , 255) , 3) # Add current fps to image

    cv2.imshow("Image" , img)
    cv2.waitKey(1)