import cv2
 
capture = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:
	ret, img = capture.read()

	#height,width,channels = img.shape
	#print( "width", width, "height: ",height)

	#img = cv2.imread("face.jpg")
	gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

	faces = face_detector.detectMultiScale(gray)
	for (x, y, w, h) in faces:
		sub_gray = gray[y:y+h, x:x+h]
		sub_color = img[y:y+h, x:x+h]
		eyes = eye_detector.detectMultiScale(sub_gray)
		if(len(eyes) > 0):
			for(ex, ey, ew, eh) in eyes:
				cv2.rectangle(sub_color, (ex, ey), (ex+ew, ey+eh), (0,0,255), 2)
			cv2.rectangle(img, (x,y), (x+w,y+h), (0,225,0), 1)

	cv2.imshow("frame", img)

	if (cv2.waitKey(1) == 27):
		break

capture.release()
cv2.destroyAllWindows
	


	
