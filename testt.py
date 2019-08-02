import cv2

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Enrollment = '156007'
Name = 'kushal'
sampleNum = 0
while (True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    print(sampleNum)
                    # saving the captured face in the dataset folder
                    cv2.imwrite("TrainingImages/ " + Name + "." + Enrollment + '.' + str(sampleNum) + ".png",
                                gray[y:y + h, x:x + w])
                    cv2.imshow('Frame', img)
                # break if the sample number is morethan 100
                if sampleNum > 200:
                    break
cam.release()
cv2.destroyAllWindows()