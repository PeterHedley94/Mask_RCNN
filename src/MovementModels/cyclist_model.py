#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures


class cycle_model:
    def __init__(self):
        self.var = 0
        # x,x_vel,x_acc,time
        self.history = np.zeros((1,8))#BayesianRidge
        self.models = {"x":linear_model.LinearRegression(),"y":linear_model.LinearRegression(),"z":linear_model.LinearRegression()}
        self.poly = PolynomialFeatures(degree=2)

    def add_points(self,pose,time):
        time = float(time)/10**9
        x,y,z = pose[1]
        dx,dy,dz = pose[2][:3]
        new_vals = np.zeros((1,8))
        if self.history.shape[0] > 1:
            since_start = time-self.history[1,6]
        else:
            since_start = 0

        new_vals[:] = x,y,z,dx,dy,dz,time,since_start
        self.history = np.concatenate([self.history,new_vals],axis=0)


    def fit(self,gap):
        features = np.zeros(((self.history.shape[0]-1)*3-3*gap,self.history.shape[1]-1))

        '''print("History is " + str(self.history))
        print("Gap is " + str(gap))'''
        for i,z in zip(range(0,features.shape[0],3),range(0,self.history.shape[0]-1,1)):
            for pert in range(3):
                i_1,i_2 = int(z+gap+pert-1),z
                f_index = i+pert
                #print("index is " + str(i_1) + " i is " + str(i) + "f_index is " + str(f_index)
                # + " index 2 " + str(i_2))
                features[f_index,-1] = self.history[i_1,-1] - self.history[i_2,-1]
                features[f_index,0] = self.history[i_1,0] - self.history[i_2,0]
                features[f_index,1] = self.history[i_1,1] - self.history[i_2,1]
                features[f_index,2] = self.history[i_1,2] - self.history[i_2,2]
        '''
        print("Features are : " + str(features[:,:]))
        print("Xs are : " + str(features[:,[0,1,2]]))
        print("Max features are : " + str(np.max(features[:-gap,3:])))
        print("Min features are : " + str(np.min(features[:-gap,3:])))
        print(" X values are : " + str(features[:,0]))'''
        vars = self.poly.fit_transform(features[3:,3:].reshape((-1,4)))
        self.models["x"].fit(vars,features[3:,0].reshape((-1,1)))
        self.models["y"].fit(vars,features[3:,1].reshape((-1,1)))
        self.models["z"].fit(vars,features[3:,2].reshape((-1,1)))


    def predict(self,time):
        #time_lapse = time - self.history[-1,6]
        if self.history.shape[0] < 3:
            return self.history[-1,[3,4,5]]*time#/10**9
        gap = 0
        for i,val in enumerate(self.history[1:,-1]):
            #print("val is " + str(val) + "time is " + str(time))
            if val >= time:
                gap = i
                break
        if gap == 0 or self.history.shape[0]-gap < 4:
            #print("Gap equals zero: returning " + str(self.history[-1,[3,4,5]]*time))
            return self.history[-1,[3,4,5]]*time#/10**9


        self.fit(gap)
        vars = self.history[-1,3:-2].tolist()
        vars.append(time)
        #print("vars are : " + str(np.array(vars).reshape((-1,4))))
        vars= self.poly.fit_transform(np.array(vars).reshape((-1,4)))
        x = self.models['x'].predict(vars) + self.history[-1,0]
        y = self.models['y'].predict(vars) + self.history[-1,1]
        z = self.models['z'].predict(vars) + self.history[-1,2]
        #print("Prediction for time " + str(time) + " is " + str([x,y,z]))
        return x,y,z
