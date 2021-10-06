import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os
import PIL

pre_train = tf.keras.models.load_model('./left_right_final.h5')
dir_path = './Testing/'
for i in os.listdir(dir_path):

    if i != ('.DS_Store'): 
      img_path = (dir_path + i)
      img = image.load_img(dir_path+i,target_size=(200,200,3))
      X = image.img_to_array(img)
      x = np.expand_dims(X,axis=0)
      images = np.vstack([x])
      val = pre_train.predict(images)
      if val == 1: 
        pre = "Detected: Right"
      else: 
        pre = "Detected: Left"
      print("Predicted Orientation is: ",pre)
      #display image and the prediction text over it
      disp_img = cv2.imread(img_path)
      #display prediction text over the image
      cv2.putText(disp_img, pre, (20,20), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (0,0,0))
      #show the image
      cv2.imshow('result',disp_img) 
      cv2.waitKey(0)
