#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
import cv2
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from Tracking.Kalman import *
from utils.common import *
from MovementModels.cyclist_rnn_model import *


class cycle_model:
    def __init__(self):
        self.im_h,self.im_w,self.im_ch = IM_HEIGHT_model, IM_WIDTH_model, NUMBER_CHANNELS
        self.data_dims = [3,4]
        self.var = 0
        self.sequence_length = 5
        self.sequence_gap = 30
        self.count = -1
        self.time_between_frames = 1.0/30#
        self.no_seq = 0
        self.max_no_seq = 1
        self.kalman = Kalman_Filter(6,3)
        self.set_up_kalman()
        self.T_WS_r = None
        self.T_WS_C = None

        #need one more than sequence length to minus from
        #probably a lot faster to convert to vector of arrays
        self.pose_history = []
        self.image_history = []
        if not USE_KALMAN_MODEL:
            self.model = rcnn_total_model(CYCLIST_MODEL_LOCATION)
        self.poly = PolynomialFeatures(degree=2)
        self.current_point = [0,0,0]



    def set_up_kalman(self):
        '''
        Fk, the state-transition model;
        Hk, the observation model;
        Qk, the covariance of the process noise;
        Rk, the covariance of the observation noise;
        Sk, Innovation (or pre-fit residual) covariance
        Yk, Innovation or measurement residual
        '''
        self.kalman.F = np.eye(6,dtype = np.float64)
        #TODO
        deltat = 1.0/30
        self.kalman.F[:3,3:] = np.diag(np.array([deltat]*3,dtype = np.float64))
        self.kalman.H = np.zeros((3,6),dtype = np.float64)
        self.kalman.H[:3,:3] = np.eye(3)
        self.kalman.errorCovPost = np.eye(6,dtype = np.float64)# 1.
        #print("Kalman number " +str(i) + " initialised with " + str(self.roi_dims_w[[0,1,2,4,5],i]))
        state = np.array([0,0,0,0,0,0])
        self.kalman.statePost = state[:,None]
        self.kalman.statePre = state[:,None]
        self.kalman.deltat = 1.0/30
        q = [10,10,10,1,1,1] #200
        r = [0.1,0.1,0.1] #2
        self.kalman.Q =  np.diag(np.array(q,dtype = np.float64))/200
        self.kalman.R = np.diag(np.array(r,dtype = np.float64))/2


    def add_points(self,pose,image,time):
        self.count += 1
        time = float(time)/10**9

        self.current_data = np.concatenate((pose[0],pose[1][:,None]),axis = 1)


        if self.count >= (self.sequence_length+1)*self.sequence_gap*self.max_no_seq:
            del self.pose_history[0]
            del self.image_history[0]
            self.count = self.count -1

        if self.count != 0 and self.count % ((self.sequence_length+1)*self.sequence_gap) == 0:
            if self.no_seq < self.max_no_seq:
                self.no_seq += 1
        self.current_point = self.current_data[:3,3:]
        self.kalman.predict()
        self.kalman.correct(self.current_point)
        self.pose_history.append(self.current_data.reshape((1,self.data_dims[0]*self.data_dims[1])))
        image = cv2.resize(image, (self.im_w,self.im_h))
        self.image_history.append((image.reshape((1,self.im_h,self.im_w,self.im_ch))-127.5)/127.5)

    def get_T_WC(self,T_WS_C,T_WS_r):
        T_WS = np.zeros((4,4),dtype = np.float64)
        T_WS[:3,:4] = np.concatenate((T_WS_C,T_WS_r[:,None]),axis = 1)
        T_WS[3,3] = 1
        T_WC = np.matmul(T_WS,self.T_SC)
        return T_WC

    def world_to_camera(self,array,T_CW):
        #array is (3,N) containing x,y,z
        N = array.shape[1]
        world_point = np.ones((4,N))
        world_point[:3,:] = array
        camera_point = np.matmul(T_CW,world_point)
        camera_point[:2,:] = camera_point[:2,:]/camera_point[2,:]
        camera_point[:2,:] = np.matmul(self.foc_mat_inv,camera_point[:2,:])
        camera_point[:3,:] = camera_point[:3,:]+self.c_mat
        return camera_point[:3,:]

    def transform_TW(self,TW_to,TW_from):
        return np.matmul(TW_to,TW_from)


    def reverse_transform(self,array):
        reversed = np.zeros((4,4),dtype = np.float64)
        reversed[:3,:4] = np.concatenate((array[:3,:3].transpose(),
                                                np.matmul(-array[:3,:3].transpose(),array[:3,3:])),axis = 1)
        reversed[3,3] = 1
        return reversed



    def fit(self):
        pose_features = np.zeros((self.no_seq,self.sequence_length+1,4*4))
        image_features = np.zeros((self.no_seq,self.sequence_length+1,self.im_h,self.im_w,self.im_ch))
        first_pose = np.zeros((self.no_seq,3))

        for sequence in range(self.no_seq):
            index = self.sequence_gap*self.sequence_length*sequence
            count = 0
            T_WC_initial = pose_features[sequence,0,:]
            T_CW_init = self.reverse_transform(T_WC_initial)

            for no_in_seq in range(1,self.sequence_length+1):
                T_W_C_next = pose_features[sequence,no_in_seq,:]
                c_ez = self.reverse_transform(T_W_C_next)[:,2].tolist()

                #TODO add c_ez
                T_W_C_transformed = self.transform_TW(T_CW_init,T_W_C_next)[:3,:]
                last_point = T_W_C_transformed[:3,3:]
                T_W_C_transformed = T_W_C_transformed.reshape((1,-1)).tolist()[0]
                T_W_C_transformed.extend(c_ez)
                pose_features[sequence,count,:] = T_W_C_transformed

                #GET IMAGES
                frame = i*self.sequence_gap + self.sequence_gap
                image_features[sequence,count,:,:,:] = self.image_history[index+frame]

            first_pose[seq_no,:] = T_WC_initial.reshape((1,-1))[:3,3:].tolist()[0]

        return pose_features,image_features,first_pose


    def predict(self):
        if USE_KALMAN_MODEL:
            return self.kalman_prediction()
        else:
            return self.model_prediction()

    def kalman_prediction(self):
        x_data,y_data,z_data,uncert = [],[],[],[]
        time_step = self.sequence_gap*self.time_between_frames/10
        for time in np.arange(0.1,2.0,0.1).tolist():

            Prediction,u = self.kalman.predict_seconds(self.sequence_gap*self.time_between_frames*time)
            x,y,z = Prediction[:3]
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)
            uncert.append(u)
        return x_data,y_data,z_data,time_step,uncert,True


    def model_prediction(self):
        if self.count < (self.sequence_length+1)*self.sequence_gap:
            return self.kalman_prediction()

        pose_features,image_features,first_pose = self.fit()
        #2 is the val loss of the network and is used as the uuncertainty
        ratio = np.arange(0.1,1.0,0.1)*2
        x,y,z = self.model.predict(pose_features,image_features)
        x += first_pose[-1,0]
        y += first_pose[-1,1]
        z += first_pose[-1,2]
        #get ten values of prediction up to future frame
        time_step = self.sequence_gap*self.time_between_frames/10
        return x,y,z,time_step,ratio,True
