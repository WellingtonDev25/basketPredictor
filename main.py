import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

video = cv2.VideoCapture('Videos/vid (3).mp4')

myColorFinder = ColorFinder(False)
hsvValues = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

pointsList = []
pointsListX = []
pointsListY = []
xList = [item for item in range(0,1300)]

while True:
    _,img = video.read()
    img = img[0:900,:] #retirar parte de baixo
    #color ball detection
    imgColor,mask = myColorFinder.update(img,hsvValues)
    #find ball location
    imgContour, contours = cvzone.findContours(img,mask,minArea=500)
    if contours:
        # cx,cy = contours[0]['center']
        pointsList.append(contours[0]['center'])
        pointsListX.append(contours[0]['center'][0])
        pointsListY.append(contours[0]['center'][1])

    if pointsListX:
        #achar o coeficiente
        coeff = np.polyfit(pointsListX,pointsListY,2)
        for point in pointsList:
            cv2.circle(imgContour,point,10,(0,255,0),cv2.FILLED)

        for x in xList:
            poly = np.poly1d(coeff)
            y = int(poly(x))
            cv2.circle(imgContour, (x,y), 2, (255, 0, 255), cv2.FILLED)
            if x >=330 and x <=410 and y >=580 and y<=610:
                #print('Dentro')
                cv2.rectangle(imgContour,(550,600),(850,660),(0,255,0),-1)
                cv2.putText(imgContour,'ACERTOU',(560,650),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),5)


    #display
    #imgColor = cv2.resize(imgColor, (0, 0), None, 0.7, 0.7)
    # cv2.imshow('Video',img)
    cv2.imshow('Video', imgContour)
    cv2.waitKey(100)