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
from utils.common import *

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
        self.class_ = class_
        self.time = time_

        self.pose = pose
        self.camera_model = camera_model
        self.remove_indices = []

        #Central point matrix [c1,c2]
        self.c_mat = np.zeros((3,1))
        self.c_mat[0,:],self.c_mat[1,:] = self.camera_model["principal_point"]
        #TRANSFORMATION TO WORLD FRAME
        self.T_WS = np.zeros((4,4),dtype = np.float64)
        self.T_WS[:3,:4] = np.concatenate((self.pose[0],self.pose[1][:,None]),axis = 1)
        self.T_WS[3,3] = 1
        self.T_WC = np.matmul(self.T_WS,np.array(self.camera_model["T_SC"]).reshape((4,4)))
        #Focal Matrix [[1/f1,0],[0,1/f2]]
        self.foc_mat = np.zeros((2,2))
        self.foc_mat[0,0],self.foc_mat[1,1] = 1/self.camera_model["focal_length"][0],1/self.camera_model["focal_length"][1]

        #TRANSFORMATION FROM WORLD FRAME
        self.T_SW = np.zeros((4,4),dtype = np.float64)
        self.T_SW[:3,:4] = np.concatenate((self.pose[0].transpose(),np.matmul(-self.pose[0].transpose(),self.pose[1][:,None])),axis = 1)
        self.T_SW[3,3] = 1

        self.T_CW = np.zeros((4,4),dtype = np.float64)
        self.T_CW[:3,:4] = np.concatenate((self.T_WC[:3,:3].transpose(),
                                                np.matmul(-self.T_WC[:3,:3].transpose(),self.T_WC[:3,3:])),axis = 1)
        self.T_CW[3,3] = 1
        #Focal Matrix Inv [[1/f1,0],[0,1/f2]]
        self.foc_mat_inv = np.zeros((2,2))
        self.foc_mat_inv[0,0],self.foc_mat_inv[1,1] = self.camera_model["focal_length"][0],self.camera_model["focal_length"][1]


        self.depth_rois = np.zeros(self.no_rois)
        self.get_depth_rois(roi_)

        #Col1-5 = centre_x,y,z,1s,width,height - camera frame
        self.roi_dims_c = np.ones((6,self.no_rois))
        #'' '' - world frame
        self.roi_dims_w = np.ones((6,self.no_rois))
        self.get_dimensions(roi_)
        self.colours = colours
        self.descriptors = des
        self.keypoints = key
        self.delete_indices()
        self.lives = [3] * self.no_rois
        self.id = np.arange(self.no_rois)
        self.object_collision = [False] * self.no_rois
        self.tracker_predictions = [0,0,0,0] * self.no_rois
        self.kalman = []
        self.tracker = []

        for i in range(self.no_rois):
            self.set_up_kalman(i)
            self.tracker.append(None)


    def intialise_tracker(self,i):
        r = tuple(self.roi_dims_c[[0,1,4,5],i])
        #print("Cdims " + str(self.roi_dims_c[[0,1,4,5],i]))
        self.tracker[i] = cv2.TrackerKCF_create()
        self.tracker[i].init(self.image,r)


    def correct_depth(self,real_height,check_size,array):
        if real_height > check_size[2] or real_height < check_size[0]:
            message = "Correcting depth measurement from " + str(array[2,0])
            c3 = self.T_WS[2,:2]
            #FROM camera_to_world
            N = array.shape[1]
            camera_point = np.ones((4,N))
            camera_point[:3,:] = array
            camera_point[:3,:] = camera_point[:3,:]-self.c_mat
            camera_point[:2,:] = np.matmul(self.foc_mat,camera_point[:2,:])
            A  = camera_point[:2,0]-camera_point[:2,1]
            depth = check_size[1]/np.matmul(c3,A)
            print(message + " to  " + str(depth))
            return depth
        else:
            return array[2,0]

    def delete_indices(self):#,roi_):
        indices = [False if x in self.remove_indices else True for x in np.arange(self.no_rois)]
        self.roi_dims_c = self.roi_dims_c[:,np.where(indices)].reshape((6,-1))
        self.roi_dims_w = self.roi_dims_w[:,np.where(indices)].reshape((6,-1))
        self.masks = self.masks[:,:,np.where(indices)].reshape(self.camera_model["image_dimension"][1],self.camera_model["image_dimension"][0],-1)
        self.class_ = self.class_[np.where(indices)]
        for i,k in enumerate(self.remove_indices):
            del self.colours[k-i]
            del self.descriptors[k-i]
            del self.keypoints[k-i]
        self.no_rois -= len(self.remove_indices)
        self.remove_indices = []
        #return roi_

    def check_depth(self,depth,bbox,i):
        #"roi[1],roi[3] - x top left, x top right"
        #"roi[0],roi[2]" - y top left, y top right
        array = np.zeros((3,2))
        array[:,0] = np.array([bbox[1],bbox[0],depth])
        array[:,1] = np.array([bbox[1],bbox[2],depth])
        world_array = self.camera_to_world(array)
        real_height = abs(world_array[2,0]-world_array[2,1])
        check_size = expected_heights[class_names[self.class_[i]]]
        depth = self.correct_depth(real_height,check_size,array)
        if depth > 13:
            self.remove_indices.append(i)
        return depth

    def get_depth_rois(self,bbox):
        for roi in range(self.no_rois):
            #print(self.masks[:,:,roi])
            depth_mask = self.depth[self.masks[:,:,roi]]
            self.depth[self.masks[:,:,roi]] = 10000000
            predicted_depth = np.median(depth_mask[depth_mask>0])/5000
            self.depth_rois[roi] = self.check_depth(predicted_depth,bbox[roi,:],roi)
            #print("predicted Depth is " + str(self.depth_rois[roi]))
            if math.isnan(self.depth_rois[roi]):
                self.depth_rois[roi] = 999


    def add_rois(self,predictions):
        self.roi_dims_w = np.concatenate([self.roi_dims_w, predictions[:6]],axis=1)
        camera_point = np.ones((6,1))
        #if kalman filter gets it wrong
        if predictions[3] <= 1:
            predictions[3] = 10

        if predictions[4] <= 1:
            predictions[4] = 10

        camera_point[:3,0,None] = self.world_to_camera(predictions[:3])
        camera_point[4:,0] = int(predictions[3]),int(predictions[4])
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
                    var = self.world_to_camera(new_state[:3].reshape((-1,1)))
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
                self.object_collision.append(False)
                self.kalman.append(older.kalman[old_])

                predictions = older.kalman[old_].statePre
                self.add_rois(predictions)

                self.tracker.append(older.tracker[old_])
                self.tracker_predictions.append([0,0,0,0])

                self.lives.append(older.lives[old_]-1)
                #self.hist.append(older.hist[old_])
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
        self.kalman[i].F = np.eye(10,dtype = np.float64)
        #TODO
        deltat = 1.0/30
        self.kalman[i].F[:5,5:] = np.diag(np.array([deltat]*5,dtype = np.float64))
        self.kalman[i].H = np.zeros((5,10),dtype = np.float64)
        self.kalman[i].H[:5,:5] = np.eye(5)
        self.kalman[i].errorCovPost = np.eye(10,dtype = np.float64)# 1.
        #print("Kalman number " +str(i) + " initialised with " + str(self.roi_dims_w[[0,1,2,4,5],i]))
        state = np.concatenate([self.roi_dims_w[[0,1,2,4,5],i],np.array([0,0,0,0,0])],axis=0)
        self.kalman[i].statePost = state[:,None]
        self.kalman[i].statePre = state[:,None]
        self.kalman[i].deltat = 1.0/30

        if class_names[self.class_[i]] == "person":
            q = [1.5,1.5,1.5,7,7,0.15,0.15,0.15,0.7,0.7]
        elif class_names[self.class_[i]] == "bicycle":
            q = [10,10,10,7,7,1,1,1,0.7,0.7]
        else:
            q = [15,15,15,7,7,1.5,1.5,1.5,0.7,0.7]

        r = [0.1,0.1,0.1,20,20]
        self.kalman[i].Q =  np.diag(np.array(q,dtype = np.float64))/200
        self.kalman[i].R = np.diag(np.array(r,dtype = np.float64))/2

    def correct_Kalman_time(self,deltat):
        for i in range(self.no_rois):
            self.kalman[i].F[:5,5:] = np.diag(np.array([deltat]*5,dtype = np.float64))


    def world_to_camera(self,array):
        #array is (3,N) containing x,y,z
        N = array.shape[1]
        world_point = np.ones((4,N))
        world_point[:3,:] = array
        camera_point = np.matmul(self.T_SW,world_point)
        camera_point[:3,:] = camera_point[:3,:]/camera_point[2,:]
        camera_point[:2,:] = np.matmul(self.foc_mat_inv,camera_point[:2,:])
        camera_point[:3,:] = camera_point[:3,:]+self.c_mat
        return camera_point[:3,:]

    def camera_to_world(self,array):
        #array is (3,N)
        N = array.shape[1]
        camera_point = np.ones((4,N))
        camera_point[:3,:] = array
        camera_point[:3,:] = camera_point[:3,:]-self.c_mat
        camera_point[:2,:] = np.matmul(self.foc_mat,camera_point[:2,:])
        camera_point[:2,:] = camera_point[:2,:]*camera_point[2,:]
        world_point = np.matmul(self.T_WS,camera_point)[:3,:]
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
        #print("T_WS is \n" + str(self.T_WS))
        #print("world_point 2 cam \n" + str(self.world_to_camera(self.roi_dims_w[:3,:])))
