# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:31:40 2019

@author: Tasha Upchurch
"""
import cv2
import numpy as np

outshape =(1024,512,3)
outshape1 =(1024,512,1)
s=5
kernel = np.ones((s,s),np.float32)/(s*s)

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

cap = cv2.VideoCapture('test.MP4')
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame',1024,512)
while(cap.isOpened()):
    ret, frame = cap.read()

    if (frame is not None):
        small = cv2.resize(frame,(256,512),interpolation= cv2.INTER_CUBIC)
        opsize = cv2.resize(frame,(512,1024),interpolation= cv2.INTER_CUBIC)
        
        gray  = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).astype('uint8')
        
        canny0 = cv2.Canny(opsize, 64,128).reshape(outshape1)
        canny1 = cv2.Canny(small, 0,64)
        canny1 = cv2.resize(canny1,(outshape1[1],outshape1[0]),interpolation= cv2.INTER_LINEAR)
        
        
        canny1 = cv2.filter2D(canny1,-1,kernel).reshape(outshape1)
        
        o = ((opsize*.33)+(canny0*.33)+(canny1*.33))
        #o=canny1
        
        
        cv2.imshow('frame',o.astype('uint8'))      
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()