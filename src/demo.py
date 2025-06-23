import math
import cv2
import time
import numpy as np
import os
import keras
from keras.layers import *
import tensorflow as tf
from mtcnn import MTCNN
from keras.models import load_model
import face_recognition as fr
import sub_functions

# Load model
model = load_model("C:\\Users\\Asus\\Desktop\\HyperFAS\\model\\fas.h5")

# Eye and mouth thresholds
ear_threshold = 0.25
mar_threshold = 0.65

def load_mtcnn_model(model_path):
    return MTCNN(model_path)

def test_one(X):
    TEMP = X.copy()
    X = (cv2.resize(X,(224,224))-127.5)/127.5
    t = model.predict(np.array([X]))[0]
    time_end=time.time()
    return t


def test_camera(mtcnn):
    cam = cv2.VideoCapture(0)
    
    total_blinks = 0
    total_mouth_open = 0
    count_eye = 0
    count_mouth = 0
    
    can_blink = False
    mouth_can_open = False
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        image = frame.copy()
        img_size = np.asarray(image.shape)[0:2]

        bounding_boxes, scores, landmarks = mtcnn.detect(image)
        
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            for det, pts in zip(bounding_boxes, landmarks):

                det = det.astype('int32')
                #print("face confidence: %2.3f" % confidence)
                det = np.squeeze(det)
                y1 = int(np.maximum(det[0], 0))
                x1 = int(np.maximum(det[1], 0))
                y2 = int(np.minimum(det[2], img_size[1]-1))
                x2 = int(np.minimum(det[3], img_size[0]-1))

                w = x2-x1
                h = y2-y1
                _r = int(max(w,h)*0.6)
                cx,cy = (x1+x2)//2, (y1+y2)//2

                x1 = cx - _r 
                y1 = cy - _r 

                x1 = int(max(x1,0))
                y1 = int(max(y1,0))

                x2 = cx + _r 
                y2 = cy + _r 

                h,w,c =frame.shape
                x2 = int(min(x2 ,w-2))
                y2 = int(min(y2, h-2))

                _frame = frame[y1:y2 , x1:x2]
                score = test_one(_frame)
                
                label = "Real" if score > 0.90 else "Fake"
                color = (0, 255, 0) if score > 0.90 else (0, 0, 255)
                
                
                
                # Extract face landmarks for eye and mouth detection
                face_landmarks = fr.face_landmarks(image, [(y1, x2, y2, x1)])
                if face_landmarks:
                    landmarks = face_landmarks[0]
                    # print("Landmarks detected:", landmarks.keys())  # Print detected facial features

                    if sub_functions.eye_close_detection(landmarks, ear_threshold):
                        count_eye += 1
                    else:
                        if count_eye >= 1:
                            total_blinks += 1
                            count_eye = 0
                            can_blink = True
                    
                    if sub_functions.mouth_open_detection(landmarks, mar_threshold):
                        count_mouth += 1
                    else:
                        if count_mouth >= 1:
                            total_mouth_open += 1
                            count_mouth = 0
                            mouth_can_open = True
                    
                    # Determine liveness
                    if can_blink and mouth_can_open:
                        label = "Real"
                        color = (0, 255, 0)
                    
                    # Display blink & mouth status
                    cv2.putText(frame, f"Blinks: {total_blinks}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, f"Mouth Open: {total_mouth_open}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Liveness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    mtcnn = load_mtcnn_model("C:\\Users\\Asus\\Desktop\\HyperFAS\\model\\mtcnn.pb")
    test_camera(mtcnn)