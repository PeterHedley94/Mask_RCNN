#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from Tracking.Kalman import *
from utils.common import *


class cycle_model:
    def __init__(self):
        self.im_h,self.im_w,self.im_ch = 640,480,3
        self.data_dims = [3,4]
        self.var = 0
        self.sequence_length = 5
        self.sequence_gap = 30
        self.count = -1
        self.time_between_frames = 1.0/30#
        self.no_seq = 0
        self.max_no_seq = 2
        self.kalman = Kalman_Filter(6,3)
        self.set_up_kalman()

        #need one more than sequence length to minus from
        #probably a lot faster to convert to vector of arrays
        self.pose_history = []#np.zeros((   self.max_no_seq*(self.sequence_length+1)*self.sequence_gap,self.data_dims[0]*self.data_dims[1]))
        self.image_history = []#np.zeros((   self.max_no_seq*(self.sequence_length+1)*self.sequence_gap,self.im_h,self.im_w,self.im_ch))
        self.models = {"x":linear_model.LinearRegression(),"y":linear_model.LinearRegression(),"z":linear_model.LinearRegression()}
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

        q = [10,10,10,1,1,1]
        r = [0.1,0.1,0.1]
        self.kalman.Q =  np.diag(np.array(q,dtype = np.float64))/100 # 1e-5 *
        self.kalman.R = np.diag(np.array(r,dtype = np.float64)) # 1e-1 *


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
        self.image_history.append((image.reshape((1,self.im_h,self.im_w,self.im_ch))-127.5)/127.5)


    def fit(self):
        pose_features = np.zeros((self.no_seq,self.sequence_length+1,self.data_dims[0]*self.data_dims[1]))
        image_features = np.zeros((self.no_seq,self.sequence_length+1,self.im_h,self.im_w,self.im_ch))
        first_pose = np.zeros((self.no_seq,self.sequence_length+1,self.data_dims[0]*self.data_dims[1]))

        count = 0
        for sequence in range(self.no_seq):
            index = self.sequence_gap*self.sequence_length*sequence
            count = 0
            for i in range(0,self.sequence_length+1):

                frame = i*self.sequence_gap + self.sequence_gap
                pose_features[sequence,count,:] = self.pose_history[index+frame]
                #3,7,11 is where the point part of the matrix is
                pose_features[sequence,count,[3,7,11]] -= self.pose_history[index+frame-self.sequence_gap][0,[3,7,11]]
                image_features[sequence,count,:,:,:] = self.image_history[index+frame]
                count += 1

        vars = self.poly.fit_transform(pose_features[:,:-1,:].reshape((-1,self.sequence_length*self.data_dims[0]*self.data_dims[1])))
        self.models["x"].fit(vars,pose_features[:,count-1,3].reshape((-1,1)))
        self.models["y"].fit(vars,pose_features[:,count-1,7].reshape((-1,1)))
        self.models["z"].fit(vars,pose_features[:,count-1,11].reshape((-1,1)))
        return pose_features,image_features


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
        if self.count < 3:
            return 0,0,0,0,0,False
        elif self.count < (self.sequence_length+1)*self.sequence_gap:
            time_step = self.sequence_gap*self.time_between_frames/10
            ratio = np.arange(0.1,1.0,0.1)
            x = (self.pose_history[-1][0,3]-self.pose_history[-2][0,3])*self.sequence_gap*ratio  + self.current_point[0]
            y = (self.pose_history[-1][0,7]-self.pose_history[-2][0,7])*self.sequence_gap*ratio  + self.current_point[1]
            z = (self.pose_history[-1][0,11]-self.pose_history[-2][0,11])*self.sequence_gap*ratio + self.current_point[2]
            return x,y,z,time_step,ratio,True

        pose_features,image_features = self.fit()
        vars = self.poly.fit_transform(pose_features[-1,1:,:].reshape((1,self.sequence_length*self.data_dims[0]*self.data_dims[1])))
        ratio = np.arange(0.1,1.0,0.1)
        x = self.models['x'].predict(vars)*ratio + self.current_point[0]
        y = self.models['y'].predict(vars)*ratio + self.current_point[1]
        z = self.models['z'].predict(vars)*ratio + self.current_point[2]
        #get ten values of prediction up to future frame
        time_step = self.sequence_gap*self.time_between_frames/10
        return x,y,z,time_step,ratio,True
