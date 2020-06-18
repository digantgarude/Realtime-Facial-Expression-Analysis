import cv2
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
thres_confidence = 0.5
prototxtPath = "./deploy.prototxt.txt"
weightsPath = "./res10_300x300_ssd_iter_140000.caffemodel"
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def detect_faces_and_emotions(frame, faceNet, emotionModel):
    faces = []
    locs = []
    preds = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # print(detections.shape)
    for i in range(0,detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > thres_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face,(48,48))
            face = img_to_array(face)
            face = np.reshape(face,[48,48,1])
            face = np.expand_dims(face, axis=0)
            face = 1/255*face
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            print(face)
            print(face.shape)
    if len(faces) > 0:
        preds = emotionModel.predict(faces)
    print("preds : ",preds)
    print("faces : ",len(faces))  
    return (locs, preds)

faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
emotionModel = load_model('./facial_1.h5')

camera = cv2.VideoCapture(0)
time.sleep(2.0)
while True:
    retVal, frame = camera.read()
    
    (locs, preds) = detect_faces_and_emotions(frame, faceNet, emotionModel)
    print(preds)
    
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        color = (0, 255, 0)
        indx_max = np.argmax(pred)
        max_pred = max(pred)
        label = labels[indx_max]+" Prob : "+str(max_pred*100)+" %"
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.imshow("Image",frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()