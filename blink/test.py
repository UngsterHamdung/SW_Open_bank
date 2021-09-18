import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import sys
import glob
import pandas as pd

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')#각점들 반환

model = load_model('models/2018_12_17_22_58_35.h5')
#model.summary()

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0) ##amin 어레이의 최소값을 반환한다
  x2, y2 = np.amax(eye_points, axis=0) #
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

# main
img_files = glob.glob('1-custombox\\001_G1\\*.jpg')
if not img_files:
      print("jpg 이미지가 없어요")
      sys.exit()

count = len(img_files)
index = 0
#while cap.isOpened():
#  ret, img_ori = cap.read()

#if not ret:
 #break
result_l = []
result_r = []
while True:
  cap = cv2.imread(img_files[index])
  if cap is None:
    print("이미지를 불러오는데 실패해썽")
    break    
  img_ori = cv2.resize(cap, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      

  faces = detector(gray)
  type(faces)

  for face in faces:
    shapes = predictor(gray, face)

    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    #cv2.imshow('l', eye_img_l) 이거하면 왼쪽 눈나옴
    #cv2.imshow('r', eye_img_r) 이거하면 오른쪽 눈나옴

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)

      # visualize
    state_l = '%.1f' if pred_l > 0.1 else '%.1f'
    state_r = '%.1f' if pred_r > 0.1 else '%.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('result', img)
    # cv2.waitKey(0)
  if float(state_l) > 0.0:
    result_l.append("1")
  else:
    result_l.append("2")
  index += 1
  
  if index >= count :
        break 

with open('result_l.txt','w',encoding='UTF-8') as f:
  for name in result_l:
        f.write(name+'\n')

# with open('result_r.txt','w',encoding='UTF-8') as f:
#   for name in result_r:
#         f.write(name+'\n')

 

  #if cv2.waitKey(1) == ord('q'):
   # break
#print(state_l)
#print(state_r)