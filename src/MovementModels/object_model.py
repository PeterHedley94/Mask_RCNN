#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
#from sklearn import linear_model
#from sklearn.preprocessing import PolynomialFeatures

class object_model:
    def __init__(self):
        self.var = 0
        # x,x_vel,x_acc,time
        self.history = np.zeros((0,7))#BayesianRidge
        #self.models = {"x":linear_model.LinearRegression(),"y":linear_model.LinearRegression(),"z":linear_model.LinearRegression()}
        #self.poly = PolynomialFeatures(degree=2)
        self.start_time = 0

    #Note time here is from dataset so must minus off previous time and nsecs -> secs
    def add_points(self,data,time):
        time = float(time)/10**9
        x,y,z = data

        if self.history.shape[0] >= 1:
            dx,dy,dz = [x,y,z] - self.history[-1,[0,1,2]]
            time_diff = time - self.start_time
            dx = dx/time_diff
            dy = dy/time_diff
            dz = dz/time_diff

        else:
            time_diff = 0
            dx,dy,dz = [0,0,0]

        self.start_time = time
        new_vals = np.zeros((1,7))
        new_vals[:] = x,y,z,dx,dy,dz,time_diff
        self.history = np.concatenate([self.history,new_vals],axis=0)

    #Note time is a value in the future i.e. in 1s
    def predict(self,time):
        #print("History is " + str(self.history))
        if self.history.shape[0] < 2:
            return self.history[-1,[0,1,2]]+ self.history[-1,[3,4,5]]*time#/10**9
        else:
            vals = self.history[-1,[0,1,2]]+self.history[:,[3,4,5]]*time
            w = np.ones(self.history.shape[0])#np.arange(self.history.shape[0])
            x,y,z = np.average(vals,axis = 0,weights = w)
            return x,y,z
