#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# Import libraries
from PIL import Image
import cv2
import sys
import numpy as np
import requests


# In[4]:


image = Image.open('image.png')
image = image.resize((450, 250))
image_arr = np.array(image)
image


# In[5]:


grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
Image.fromarray(grey)


# In[6]:


blur = cv2.GaussianBlur(grey,(5,5),0)
Image.fromarray(blur)


# In[7]:


dilated = cv2.dilate(blur, np.ones((3,3)))
Image.fromarray(dilated)


# In[8]:


kernle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernle)
Image.fromarray(closing)


# In[10]:


car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)


# In[11]:


cnt = 0
for (x,y,w,h) in cars:
    cv2.rectangle(image_arr, (x,y),(x+w,y+h),(255,0,0),2)
    cnt += 1
print(cnt, " cars found ")
Image.fromarray(image_arr)


# In[13]:


cascade_src = '/kaggle/input/vehicle-detect-count/cars.xml'
video_src = '/kaggle/input/vehicle-detect-count/Cars.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
video = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (450,250))


# In[14]:


while True:
    ret, img = cap.read()
    
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        
    video.write(img)
video.release()


# In[ ]:




