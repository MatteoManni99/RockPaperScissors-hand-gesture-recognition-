import cv2
import joblib
import pandas as pd
import mediapipe as mp
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands = 1, min_detection_confidence = 0.7)
mpDraw = mp.solutions.drawing_utils

def img_to_pixel(img, resize_x = 25, resize_y = 25):
    new_size = (resize_x, resize_y)
    resized_img = cv2.resize(img, new_size)

    # Convert the image to greyscale
    grey_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    np_array = grey_img.reshape(1, resize_x * resize_y)
    return np_array


def img_to_landmarks(image):
    results = hands.process(image)
    if results.multi_hand_landmarks:
        landmarks_list = []
        for num, lm in enumerate(results.multi_hand_landmarks[0].landmark):
            x, y = lm.x, lm.y
            landmarks_list.append(x)
            landmarks_list.append(y)
        np_array = np.asarray(landmarks_list)
        np_array = np_array.reshape(1, 42)
        return np_array

def img_to_distance(image):
    results = hands.process(image)

    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0]
        distance_list = []
        points = [(4, 2), (5, 8), (9, 12), (13, 16), (17, 20)]
        for a, b in points:
            dx = pow((lms.landmark[a].x) - (lms.landmark[b].x), 2)
            dy = pow((lms.landmark[a].y) - (lms.landmark[b].y), 2)
            d = pow((dx + dy), 0.5)
            distance_list.append(d)
        np_array = np.asarray(distance_list).reshape(1, 5)
        return np_array

vid = cv2.VideoCapture(0)

corp_x = 0
corp_y = 0
corp_h = 500
corp_w = 500

model_selected = 'landmarks'

if model_selected == 'distances':
    clf = joblib.load('models/knn_distances.pkl')
    while True:
        ret, frame = vid.read()
        frame = frame[corp_y:corp_y + corp_h, corp_x:corp_x + corp_w]
        cv2.imshow('frame', frame)
        np_array = pd.DataFrame(img_to_distance(frame))
        if not np_array.empty:
            print(clf.predict(np_array))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if model_selected == 'landmarks':
    clf = joblib.load('models/rf_aug_landmarks.pkl')
    while True:
        ret, frame = vid.read()
        frame = frame[corp_y:corp_y + corp_h, corp_x:corp_x + corp_w]
        cv2.imshow('frame', frame)
        np_array = pd.DataFrame(img_to_landmarks(frame))
        if not np_array.empty:
            print(clf.predict(np_array))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()