import cv2
frameWidth = 640
frameHeight = 480
numPlateCascade= cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea=500
color=(255,0,255)#color in BGR format
cap = cv2.VideoCapture(1)#this will capture the webcam feed [1 for external webcam and 0 for default inbuilt webcam]
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)#this will adjust the brightness of our webcam to 150 [id will always be 10]
while True:
    success, img = cap.read()#our image would be saved in 'img' and 'success' will tell us whether it was done successfully or not.
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converts our original image to a gray image
    numPlates = numPlateCascade.detectMultiScale(imgGray,1.1,4)#here 1.1 is our scale factor and 4 is our min no. of neighbors
    for (x,y,w,h) in numPlates:
        area=w*h
        if area > minArea:#this loop will only work when our area > 500
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#creates the bounding box
            cv2.putText(img, "Number Plate",(x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,color,2)
            imgROI = img[y:y+h,x:x+w]#this will give us the region of our no. plate
            cv2.imshow("ROI", imgROI)
    
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
