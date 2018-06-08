# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 00:30:33 2017

@author: NB-SH-001
"""

import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
#    cv2.imshow('frame',gray)
#    cv2.imshow('frame',frame)
    cv2.imshow('image',cv2.resize(frame,(320,240)))    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

