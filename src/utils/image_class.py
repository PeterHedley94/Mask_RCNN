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

    def remove(self):
        try:
            return self.buffer.pop()
        except IndexError:
            return False

class roi_class:
    #removed features
    def __init__(self,roi_,time_,class_,colours,mask,des,key,image,depth,pose,camera_model):

        #self.roi = roi_
        self.no_rois = roi_.shape[0]

        self.depth = depth
        self.image = image
        self.masks = mask

        self.pose = pose
        self.camera_model = camera_model
        #Focal Matrix [[1/f1,0],[0,1/f2]]
        self.foc_mat = np.zeros((2,2))
        self.foc_mat[0,0],self.foc_mat[1,1] = 1/self.camera_model["focal_length"][0],1/self.camera_model["focal_length"][1]
        #Central point matrix [c1,c2]
        self.c_mat = np.zeros((3,1))
        self.c_mat[1,:],self.c_mat[0,:] = self.camera_model["principal_point"]
        #TRANSFORMATION TO WORLD FRAME
        self.T_Ws = np.concatenate((self.pose[0],self.pose[1][:,None]),axis = 1)

        self.T_SW = np.concatenate((self.pose[0].transpose(),np.matmul(self.pose[0].transpose(),self.pose[1][:,None])),axis = 1)
        #Focal Matrix Inv [[1/f1,0],[0,1/f2]]
        self.foc_mat_inv = np.zeros((2,2))
        self.foc_mat_inv[0,0],self.foc_mat_inv[1,1] = self.camera_model["focal_length"][0],self.camera_model["focal_length"][1]


        self.time = time_

        #Col1-5 = centre_x,y,z,1s,width,height - camera frame
        self.roi_dims_c = np.ones((6,self.no_rois))
        #'' '' - world frame
        self.roi_dims_w = np.ones((6,self.no_rois))
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

    def intialise_tracker(self,i):
        r = tuple(self.roi_dims_c[[0,1,4,5],i])
        self.tracker[i] = cv2.TrackerKCF_create()
        self.tracker[i].init(self.image,r)

    def get_depth_rois(self):
        for roi in range(self.no_rois):
            #print(self.masks[:,:,roi])
            depth_mask = self.depth[self.masks[:,:,roi]]
            self.depth[self.masks[:,:,roi]] = 10000000
            self.depth_rois[roi] = np.mean(depth_mask[depth_mask>0])/5000
            #print("predicted Depth is " + str(self.depth_rois[roi]))
            if math.isnan(self.depth_rois[roi]):
                self.depth_rois[roi] = 999


    def add_rois(self,predictions):
        self.roi_dims_w = np.concatenate([self.roi_dims_w, predictions[:6]],axis=1)
        camera_point = np.ones((6,1))
        camera_point[:3,0,None],camera_point[4:,0,None] = self.world_to_camera(predictions[:3]),predictions[3:5]
        self.roi_dims_c = np.concatenate([self.roi_dims_c,camera_point],axis=1)
        self.depth_rois = np.concatenate((self.depth_rois,camera_point[2,0,None]), axis=0)

    def id_new_objects(self,final_indices):
        if len(self.id) > 0:
            max_id = max(self.id)+1
        else:
            max_id = 0
        #Add objects that only appear in new frame
        for c,index in enumerate(final_indices):
            if index == -1:
                self.id[c] = max_id
                max_id += 1

        if len(self.id) != len(set(self.id)):
            print("THERE ARE DUPLICATE IDS")


    def match_old_rois(self,older,final_indices):
        #Add objects detected in both frames
        for c,index in enumerate(final_indices):
            if index < len(older.id) and index >= 0:
                self.id[c] = older.id[index]
                self.colours[c] = older.colours[index]

                #UPDATE KALMAN
                new_state = np.array(self.roi_dims_w[[0,1,2,4,5],c])
                kll = older.kalman[index].get_log_likelihood(new_state[:,None])

                if kll < 10**6:
                    self.kalman[c] = older.kalman[index]
                    self.kalman[c].correct(new_state)
                else:
                    print("Matched with large kll of : " + str(kll))

                if older.lives[index] < 7:
                    self.lives[c] = older.lives[index] +1


    #Add older ROIs that were not found in current frame
    def append_rois(self,older,not_in):
        no_added = 0
        for old_ in not_in:
            if older.lives[old_]-1 > 0:
                no_added += 1
                self.id = np.concatenate((self.id,older.id[old_,None]), axis=0)
                self.class_ = np.concatenate((self.class_,older.class_[old_,None]),axis = 0)
                self.kalman.append(older.kalman[old_])

                predictions = older.kalman[old_].statePre
                self.add_rois(predictions)

                self.tracker.append(older.tracker[old_])
                self.tracker_predictions.append([0,0,0,0])

                self.lives.append(older.lives[old_]-1)
                self.hist.append(older.hist[old_])
                self.colours.append(older.colours[old_])
                self.descriptors.append(older.descriptors[old_])
                self.keypoints.append(older.keypoints[old_])

                #TODO some extra logic to transform mask to other camera position
                if self.masks.shape[2] == 0:
                    self.masks = older.masks[:,:,old_,None]
                else:
                    self.masks = np.concatenate([self.masks,older.masks[:,:,old_,None]],axis = 2)
        self.no_rois += no_added


    def set_up_kalman(self,i):
        '''
        Fk, the state-transition model;
        Hk, the observation model;
        Qk, the covariance of the process noise;
        Rk, the covariance of the observation noise;
        Sk, Innovation (or pre-fit residual) covariance
        Yk, Innovation or measurement residual
        '''
        self.kalman.append(Kalman_Filter(10,5))
        self.kalman[i].F = np.eye(10)
        deltat = 1.0
        self.kalman[i].F[:5,5:] = np.diag(np.array([deltat]*5,dtype = np.float64))
        self.kalman[i].H = np.zeros((5,10))
        self.kalman[i].H[:5,:5] = np.eye(5)
        self.kalman[i].Q =  np.diag(np.array([1,1,1,20,20,0,0,0,0,0],dtype = np.float64)) # 1e-5 *
        self.kalman[i].R = np.diag(np.array([1,1,1,20,20],dtype = np.float64)) # 1e-1 *
        self.kalman[i].errorCovPost = np.eye(10)# 1.
        state = np.concatenate([self.roi_dims_w[[0,1,2,4,5],i],np.array([0,0,0,0,0])],axis=0)
        self.kalman[i].statePost = state[:,None]

    def correct_Kalman_time(self,deltat):
        for i in range(self.no_rois):
            self.kalman[i].F[:5,5:] = np.diag(np.array([deltat]*5,dtype = np.float64))


    def world_to_camera(self,array):
        #array is (3,N) containing x,y,z
        N = array.shape[1]
        world_point = np.ones((4,N))
        world_point[:3,:] = array
        camera_point = np.matmul(self.T_SW,world_point)
        camera_point[:2,:] = np.matmul(self.foc_mat_inv,camera_point[:2,:])
        camera_point[:3,:] = camera_point[:3,:]+self.c_mat
        return camera_point

    def camera_to_world(self,array):
        #array is (3,N)
        N = array.shape[1]
        camera_point = np.ones((4,N))
        camera_point[:3,:] = array
        camera_point[:3,:] = camera_point[:3,:]-self.c_mat
        camera_point[:2,:] = np.matmul(self.foc_mat,camera_point[:2,:])
        world_point = np.matmul(self.T_Ws,camera_point)
        return world_point


    def get_dimensions(self,roi):
        #"roi[1],roi[3] - x top left, x top right"
        #"roi[0],roi[2]" - y top left, y top right
        self.roi_dims_c[0,:] = ((roi[None,:,1]+roi[None,:,3])/2)[0]
        self.roi_dims_c[1,:] = ((roi[None,:,0] + roi[None,:,2])/2)[0]
        self.roi_dims_c[2,:] = self.depth_rois
        self.roi_dims_c[4,:] = (roi[None,:,3] - roi[None,:,1])[0]
        self.roi_dims_c[5,:] = (roi[None,:,2] - roi[None,:,0])[0]

        self.roi_dims_w[:3,:] = self.camera_to_world(self.roi_dims_c[:3,:])
        self.roi_dims_w[3:,:] = self.roi_dims_c[3:,:]

        #print("camera_point \n" + str(self.roi_dims_c[:3,:]))
        #print("world_point \n" + str(self.roi_dims_w[:3,:]))
