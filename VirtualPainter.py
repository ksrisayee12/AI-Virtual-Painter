import cv2 as cv
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 20
eraserThickness = 60

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
xp, yp =0, 0

imageCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1.Import The Image
    success, img = cap.read()
    img = cv.flip(img, 1)

    # 2.Finding Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPositions(img, draw=False)

    if len(lmList)!=0:
        # print (lmList)
        # Tip Of Index And Middle Fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3.Check Which Fingers Are Up
        fingers = detector.fingersUp()
        # print(fingers)


        # 4.If Selection Mode - Two Fingers Are Up
        if fingers[1] and fingers[2]:
            xp, yp =0, 0
            # print('Selection Mode')
            # Checking For Changing Colors
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv.rectangle(img, (x1, y1-35), (x2, y2+35), drawColor, cv.FILLED)

        # 5.If Drawing Mode - Index Finger Should Be Up
        if fingers[1] and fingers[2] == False:
            cv.circle(img, (x1, y1), 20, drawColor, cv.FILLED)
            # print('Drawing Mode')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            
            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imageCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv.line(imageCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv.cvtColor(imageCanvas, cv.COLOR_BGR2GRAY)
    _, imgInverse = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInverse = cv.cvtColor(imgInverse, cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInverse)
    img = cv.bitwise_or(img, imageCanvas)
    
    # Setting The header Image
    img[0:125, 0:1280] = header
    # img = cv.addWeighted(img, 0.5, imageCanvas, 0.5, 0)
    cv.imshow('PAINTING AREA', img)
    # cv.imshow('Canvas', imageCanvas)
    cv.waitKey(1)
