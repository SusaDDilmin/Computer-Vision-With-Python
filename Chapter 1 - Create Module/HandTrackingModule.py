import cv2 
import mediapipe as mp
import time

# Create a Class
class handDetector():
    def __init__(self, mode=False, maxHands=2,model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Hand Tracking - Using MediaPipe (like starting point)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.model_complexity ,self.detectionCon, self.trackCon) # detection of hands
        self.mpDraw = mp.solutions.drawing_utils # drawing landmarks

    def findHands(self , img , draw = True):
        #img is in BGR - Convert it to RGB
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) # process the image - detection of hands
        #print(results.multi_hand_landmarks) #check if hands are detected

        if self.results.multi_hand_landmarks: #Check whether hands are detected
            for handLms in self.results.multi_hand_landmarks: #draw landmarks on each hand detected
                if draw:
                    self.mpDraw.draw_landmarks(img , handLms , self.mpHands.HAND_CONNECTIONS) # draw connections & landmarks
                    
        return img
    
    #Can use to draw circle on landmarks of specific hand
    def findPosition(self , img , handNo = 0 , draw = True):
        lmList = [] # list to store landmarks
        if self.results.multi_hand_landmarks:   # Check whether hands are detected
            myHand = self.results.multi_hand_landmarks[handNo]  # Select What hand to detect
            for id , lm in enumerate(myHand.landmark): # enumerate to get id of each landmark
                #print(id , lm) # print id of each landmark
                h , w , c = img.shape # get height , width , channels of image - using x , y coordinates
                cx , cy = int(lm.x * w) , int(lm.y * h) # get x , y coordinates of each landmark
                #print(id , cx , cy)
                lmList.append([id , cx , cy])
                if draw: # Select 0th landmark
                    cv2.circle(img , (cx , cy) , 15 , (255 , 0 , 255) , cv2.FILLED) # draw circle on 0th landmark
        return lmList
        

                
# Main Function - code inside this function can be used else where and can use this handDetector as a module , Have to import
# look at UseModuleExample.py
def main():
    #for fps
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time() #Get current time
        fps = 1/(cTime - pTime) #calculate frames per second
        pTime = cTime 

        cv2.putText(img , str(int(fps)) , (10 , 70) , cv2.FONT_HERSHEY_PLAIN , 3 , (255 , 0 , 255) , 3) # Add current fps to image

        cv2.imshow("Image" , img)
        cv2.waitKey(1)

# If run this py file directly, it will run the below code
if __name__ == "__main__":
    main()