import cv2
import os
from PIL import Image
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()


    def face_detect(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        sampleNum = 0
        while True:
            ret, img = self.video.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()

    sampleNum = 0

    def get_frame(self,Name,Enrollment):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        while True:
            ret, img = self.video.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if self.sampleNum > 50:
                break

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
                self.sampleNum +=  1
                print(self.sampleNum)
                cv2.imwrite("TrainingImages/ " + Enrollment + "." + Name + '.' + str(self.sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])

                # saving the captured face in the dataset folder

                ret, jpeg = cv2.imencode('.jpg', img)
                return jpeg.tobytes()
            cv2.destroyAllWindows()





def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids