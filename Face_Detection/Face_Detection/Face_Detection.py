import cv2 as cv

#Load Pre-trained Data on face frontals from open source cv (haarcascade)
trained_face_data = cv.CascadeClassifier('../Trained Dataset/haarcascade_frontalface_default.xml')

#Way to feed in the vdo/image

img = cv.imread('./Dataset/rdj.jpg')

#show Image
#cv.imshow('Vibhu Face Detector' , img)

#Convert into grayScale
grayScale_img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

#cv.imshow('grey',grayScale_img)





#Waits till any key is pressed
cv.waitKey()



#on Successful Compilation
print("Code Done")