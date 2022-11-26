import cv2
from cvzone.ColorModule import ColorFinder

img = cv2.imread('Ball.png')
colorFinder = ColorFinder(True)

while True:
    imgColor,mask = colorFinder.update(img)

    cv2.imshow('IMG',imgColor)
    cv2.waitKey(1)



