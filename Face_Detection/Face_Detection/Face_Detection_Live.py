
import cv2 as cv

#Load Pre-trained Data on face frontals from open source cv (haarcascade)
trained_face_data = cv.CascadeClassifier('../Trained Dataset/haarcascade_frontalface_default.xml')

webcam = cv.VideoCapture(0)


#Iterate Frame Wise

while True:

    #Read curr Frame
    successfulRead , frame = webcam.read()
    
    #GrayScaling
    grayScale_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    #cv.imshow('Face Detector Live', grayScale_frame)
    face_Coordinates = trained_face_data.detectMultiScale(grayScale_frame)

    for (x,y,w,h) in face_Coordinates:
        cv.rectangle(frame,(x,y) , (x+w,y+h) , (0,255,0),2)
    cv.imshow('LIVE FACE DETECTION',frame)

    cv.waitKey(1)
