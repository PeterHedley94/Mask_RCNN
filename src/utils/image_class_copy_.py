#!/usr/bin/env python
import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
import collections,time,math
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
    def __init__(self,roi_,time_,class_,colours,mask,des,key,image,depth,pose,camera_model):

        self.camera_model = camera_model
        self.depth = depth
        self.image = image
        self.masks = mask
        self.pose = pose
        self.time = time_
        self.roi = roi_
        self.no_rois = roi_.shape[0]
        #Col1-5 = centre_x,y,z,1s,width,height - camera frame
        self.roi_dims_c = np.ones((self.no_rois,6))
        #'' '' - world frame
        self.roi_dims_w = np.ones((self.no_rois,6))
        self.centre_x = np.zeros(self.no_rois)
        self.centre_y = np.zeros(self.no_rois)
        self.roi_width = np.zeros(self.no_rois)
        self.roi_height = np.zeros(self.no_rois)
        self.depth_rois = np.zeros(self.no_rois)
        self.get_depth_rois()
        self.get_dimensions(roi_)


        self.hist = [None] * self.no_rois
        self.id = np.arange(self.no_rois)
        self.colours = colours
        self.lives = [3] * self.no_rois
        self.class_ = class_
        self.descriptors = des
        self.keypoints = key

        self.tracker_predictions = [0,0,0,0] * self.no_rois
        self.kalman = []
        self.tracker = []


        for i in range(self.no_rois):
            self.set_up_kalman(i)
            self.tracker.append(None)
            r = (self.centre_x[i],self.centre_y[i],self.roi_width[i],self.roi_height[i])

    def intialise_tracker(self,i):
        r = (self.centre_x[i],self.centre_y[i],self.roi_width[i],self.roi_height[i])
        self.tracker[i] = cv2.TrackerKCF_create()
        self.tracker[i].init(self.image,r)

    def get_depth_rois(self):
        for roi in range(self.no_rois):
            depth_mask = self.depth[np.where(self.masks[:,:,roi]>0)]
            self.depth_rois[roi] = np.mean(depth_mask[depth_mask>0])/5000
            if math.isnan(self.depth_rois[roi]):
                self.depth_rois[roi] = 999
            print("PREDICTED DEPTH IS " + str(self.depth_rois[roi]))

    def add_rois(self,predictions):
        self.centre_x.append(predictions[0])
        self.centre_y.append(predictions[1])
        self.roi_width.append(predictions[2])
        self.roi_height.append(predictions[3])


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

    def get_dimensions(self,roi):
        #"roi[1],roi[3] - x top left, x top right"
        #"roi[0],roi[2]" - y top left, y top right
        self.roi_dims_c[:,0] = ((roi[None,:,1]+roi[None,:,3])/2)[0]
        self.roi_dims_c[:,1] = ((roi[None,:,0] + roi[None,:,2])/2)[0]
        self.roi_dims_c[:,2] = self.depth_rois
        self.roi_dims_c[:,4] = (roi[None,:,3] - roi[None,:,1])[0]
        self.roi_dims_c[:,5] = (roi[None,:,2] - roi[None,:,0])[0]

        T_W = np.concatenate((self.pose[0],self.pose[1][:,None]),axis = 1)
        print(T_W)
        print(self.roi_dims_c[:,:4])
        w,h = self.depth.shape


        foc_mat = np.zeros((2,2))
        foc_mat[0,0],foc_mat[1,1] = 1/self.camera_model["focal_length"][0],1/self.camera_model["focal_length"][1]
        c_mat = np.zeros((self.no_rois,2))
        c_mat[:,1],c_mat[:,0] = self.camera_model["principal_point"]

        print("Camera View " + str((self.roi_dims_c[:,:2]-c_mat).transpose()))
        print("Camera View Pt2 " + str(np.matmul(foc_mat,(self.roi_dims_c[:,:2]-c_mat).transpose())))

        self.roi_dims_w[:,:2] = np.matmul(foc_mat,(self.roi_dims_c[:,:2]-c_mat).transpose()).transpose()
        self.roi_dims_w[:,:3] = np.matmul(T_W,self.roi_dims_w[:,:4].transpose()).transpose()
        self.roi_dims_w[:,3:] = self.roi_dims_c[:,3:]
        print(self.roi_dims_w[:,:3])
        print("...............................................................................")
        self.centre_x = list(self.roi_dims_c[:,0])
        self.centre_y = list(self.roi_dims_c[:,1])
        self.roi_width = list(self.roi_dims_c[:,4])
        self.roi_height = list(self.roi_dims_c[:,5])
