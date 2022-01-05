import cv2 as cv

#Load Pre-trained Data on face frontals from open source cv (haarcascade)
trained_face_data = cv.CascadeClassifier('../Trained Dataset/haarcascade_frontalface_default.xml')

#Way to feed in the vdo/image

img = cv.imread('../Dataset//rdj.jpg')





#show Image
#cv.imshow('Vibhu Face Detector' , img)

#Convert into grayScale
grayScale_img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

#detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayScale_img)
#Draw Rectangle (244 208 410 410)
#(x,y,w,h) = face_coordinates[0]

for(x,y,w,h) in face_coordinates:
    cv.rectangle(img, (x,y),(x+w,y+h) , (0,255,0)  ,2)


print(face_coordinates)
cv.imshow('Face Detector',img)







#Waits till any key is pressed
cv.waitKey()



#on Successful Compilation
print("Code Done")