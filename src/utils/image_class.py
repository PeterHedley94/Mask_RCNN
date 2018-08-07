#!/usr/bin/env python
import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
import collections,time
import numpy as np
import cv2
from Tracking.Kalman import *
from sensor_msgs.msg import Image
class image_class:
    def __init__(self,frame_,d_,time):
        self.time = time
        self.frame = frame_
        self.depth = d_

class circular_buffer:
    #Creating a circular buffer for pictures
    def __init__(self):
        self.buffer = collections.deque(maxlen=2)

    def insert(self,image):
        self.buffer.append(image)
        #print("Buffer size is now " + str(len(self.image_buffer)))

    def remove(self):
        try:
            return self.buffer.pop()
        except IndexError:
            return False

class roi_class:
    #removed features
    def __init__(self,roi_,time_,class_,colours,mask,des,key,image):

        self.time = time_
        self.roi = roi_
        self.hist = [None] * roi_.shape[0]
        self.id = np.arange(roi_.shape[0])
        self.colours = colours
        self.masks = mask
        self.image = image
        #self.features = features
        self.lives = [3] * roi_.shape[0]
        self.class_ = class_
        self.descriptors = des
        self.keypoints = key
        self.centre_x = np.zeros(roi_.shape[0])
        self.centre_y = np.zeros(roi_.shape[0])
        self.roi_width = np.zeros(roi_.shape[0])
        self.roi_height = np.zeros(roi_.shape[0])
        self.get_dimensions()
        self.tracker_predictions = [0,0,0,0] * roi_.shape[0]
        self.kalman = []
        self.tracker = []


        for i in range(roi_.shape[0]):
            self.set_up_kalman(i)
            self.tracker.append(None)
            r = (self.centre_x[i],self.centre_y[i],self.roi_width[i],self.roi_height[i])

    def intialise_tracker(self,i):
        r = (self.centre_x[i],self.centre_y[i],self.roi_width[i],self.roi_height[i])
        self.tracker[i] = cv2.TrackerKCF_create()
        self.tracker[i].init(self.image,r)

    def set_up_kalman(self,i):
        '''
        Fk, the state-transition model;
        Hk, the observation model;
        Qk, the covariance of the process noise;
        Rk, the covariance of the observation noise;
        Sk, Innovation (or pre-fit residual) covariance
        Yk, Innovation or measurement residual
        '''
        self.kalman.append(Kalman_Filter(8,4))
        self.kalman[i].F = np.eye(8)
        deltat = 1.0
        self.kalman[i].F[:4,4:] = np.diag(np.array([deltat]*4,dtype = np.float64))
        self.kalman[i].H = np.zeros((4,8))
        self.kalman[i].H[:4,:4] = np.eye(4)
        self.kalman[i].Q =  np.diag(np.array([5,5,5,5,0,0,0,0],dtype = np.float64)) # 1e-5 *
        self.kalman[i].R = np.diag(np.array([5,5,20,20],dtype = np.float64)) # 1e-1 *
        self.kalman[i].errorCovPost = np.eye(8)# 1.
        state = np.array([self.centre_x[i],self.centre_y[i],self.roi_width[i],self.roi_height[i],0,0,0,0])
        self.kalman[i].statePost = state[:,None]

    def correct_Kalman_time(self,deltat):
        for i in range(self.roi_.shape[0]):
            self.kalman[i].F[:4,4:] = np.diag(np.array([deltat]*4,dtype = np.float64))

    def get_dimensions(self):
        #"roi[1],roi[3] - x top left, x top right"
        #"roi[0],roi[2]" - y top left, y top right
        self.centre_x = list(((self.roi[None,:,1]+self.roi[None,:,3])/2)[0])
        self.centre_y = list(((self.roi[None,:,0] + self.roi[None,:,2])/2)[0])
        self.roi_width = list((self.roi[None,:,3] - self.roi[None,:,1])[0])
        self.roi_height = list((self.roi[None,:,2] - self.roi[None,:,0])[0])
