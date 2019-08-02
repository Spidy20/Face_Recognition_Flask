from flask import Flask, render_template, Response, jsonify,request
from camera import VideoCamera
import cv2
from camera import getImagesAndLabels
import numpy as np

app = Flask(__name__)

video_stream = VideoCamera()

@app.route('/Attendance_Management_System')
def home():
    return render_template('home.html')

@app.route('/detection')
def index():
    return render_template('face_detect.html')

@app.route('/train_model' )
def train_model():
    return render_template('train_m.html')

@app.route('/get_image', methods=['POST'] )
def get_image():
    global en,name
    en = request.form['en']
    name = request.form['st']
    print(en,name)
    return render_template('frame_data.html',nm=name,enr = en)

@app.route('/training')
def trainimg():
    cv2.destroyAllWindows()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detector
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    try:
        global faces,Id
        faces, Id = getImagesAndLabels("TrainingImages")
    except Exception as e:
        print(e)
    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save("TrainingImageLabel\Trainner.yml")
    except Exception as e:
        print(e)
    return render_template('training.html')


def gen(camera):
    while True:
        frame = camera.get_frame(en,name)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
        return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")