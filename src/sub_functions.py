import face_recognition
from scipy.spatial import distance as dist
import os
import math

# check eye close or not
def eye_close_detection(face_landmarks, ear_threshold):
    if not face_landmarks:  # Check if landmarks were detected
        return False
    
    left_eye = face_landmarks.get('left_eye')
    right_eye = face_landmarks.get('right_eye')

    if not left_eye or not right_eye:
        return False

    ear_left = get_ear(left_eye)
    ear_right = get_ear(right_eye)

    return ear_left <= ear_threshold and ear_right <= ear_threshold


# calculate eye aspect ratio
def get_ear(eye):

    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# 找到图像中(面积)最大的脸
def find_main_face(face_locations):
    max_area = 0
    max_face = face_locations[0]
    max_face_location = []
    for face in face_locations:
        area = abs((face[0] - face[2]) * (face[1] - face[3]))
        if area > max_area:
            max_area = area
            max_face = face

    max_face_location.append(max_face)

    return max_face_location


# 人脸匹配，返回对应的名字的下标
def recognition(known_face_encodings, main_face_encoding):
    matches = face_recognition.compare_faces(known_face_encodings, main_face_encoding[0])

    if True in matches:
        index = matches.index(True)
        return index
    else:
        return None


# 判断是否张嘴
def mouth_open_detection(face_landmarks, mar_threshold):
    if not face_landmarks:  # Check if landmarks were detected
        return False

    top_lip = face_landmarks['top_lip']
    bottom_lip = face_landmarks['bottom_lip']

    mouth_mar = check_mouth_open(top_lip,bottom_lip,mar_threshold)
    return mouth_mar


def get_lip_height(lip):
    sum=0
    for i in [2,3,4]:
        # distance between two near points up and down
        distance = math.sqrt( (lip[i][0] - lip[12-i][0])**2 +
                              (lip[i][1] - lip[12-i][1])**2   )
        sum += distance
    return sum / 3

def get_mouth_height(top_lip, bottom_lip):
    sum=0
    for i in [8,9,10]:
        # distance between two near points up and down
        distance = math.sqrt( (top_lip[i][0] - bottom_lip[18-i][0])**2 + 
                              (top_lip[i][1] - bottom_lip[18-i][1])**2   )
        sum += distance
    return sum / 3

def check_mouth_open(top_lip, bottom_lip,ratio):
    top_lip_height =    get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height =      get_mouth_height(top_lip, bottom_lip)

    # if mouth is open more than lip height * ratio, return true.
    if mouth_height > min(top_lip_height, bottom_lip_height) * ratio:
        return True
    else:
        return False


# 从文件夹中加载图片
def load_known_persons(path):
    known_faces_encodings = []
    known_faces_names = []

    for roots, dirs, files in os.walk(path):
        for file in files:
            file_fullname = os.path.join(roots, file)
            img = face_recognition.load_image_file(file_fullname)
            face_encoding = face_recognition.face_encodings(img)[0]
            known_faces_encodings.append(face_encoding)
            name = file.split('.')[0]
            known_faces_names.append(name)

    return known_faces_encodings, known_faces_names

