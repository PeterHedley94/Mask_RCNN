#!/usr/bin/env python
import rospy
import sys,threading
import time,math
from collections import deque
print(sys.version)
from std_msgs.msg import String
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np,imutils
import pandas as pd
from sklearn import linear_model
#import Path


class dataframe_:
    def __init__(self,columns,start_length):
        self.dataframe = pd.DataFrame(index=np.arange(start_length), columns=columns)
        self.length = 0
        self.space = start_length

    def add_data(self,data):
        self.dataframe.iloc[self.length] = data
        if self.length > self.space/2:
            self.dataframe = 1


class pose_visualiser:
    def __init__(self):
        self.xpoints = []
        self.ypoints = []
        self.zpoints = []
        self.T_WS_C = []
        self.T_WS_r = 0
        self.max_x = 0
        self.max_y = 0
        self.max_z = 0
        self.pointcloud = []
        self.prediction = [0,0,0]
        self.models = {"x":linear_model.BayesianRidge(),"y":linear_model.BayesianRidge(),"z":linear_model.BayesianRidge()}
        self.model_colnames = ["dx","dy","dz","d2x","d2y","d2z"]
        self.model_data = pd.DataFrame(columns = self.model_colnames)
        self.locks = {"position":threading.Lock(),'pointcloud':threading.Lock()}
        self.max_time_secs = 0
        self.max_time_nsecs = 0
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection="3d")
        #plt.show()


    def linear_model_predict(self):
        """Very pointless but simple model"""
        #x = np.array(self.model_data["dx"])
        #y = np.array(self.model_data["dy"])
        #z = np.array(self.model_data["dz"])
        #vals = np.array(self.model_data.drop(["dx","dy","dz"],axis=1))

        self.models["x"].fit(np.array(self.model_data["d2x"]).reshape(-1, 1),self.model_data["dx"])
        self.models["y"].fit(np.array(self.model_data["d2y"]).reshape(-1, 1),self.model_data["dy"])
        self.models["z"].fit(np.array(self.model_data["d2z"]).reshape(-1, 1),self.model_data["dz"])


        #vals = np.zeros((1,6))
        #vals[0,:] = [self.xpoints[-1],self.ypoints[-1],self.zpoints[-1],self.xpoints[-1]-self.xpoints[-2],self.ypoints[-1]-self.ypoints[-2],self.zpoints[-1]-self.zpoints[-2]]
        self.prediction[0] = self.models['x'].predict(np.array(self.model_data["dx"])[0])
        self.prediction[1] = self.models['y'].predict(np.array(self.model_data["dy"])[0])
        self.prediction[2] = self.models['z'].predict(np.array(self.model_data["dz"])[0])
        print("predictions are : " + str(self.prediction))


    def pose_callback(self,depth,T_WS_r,T_WS_C,camera_model,w,h):

        self.xpoints.append(T_WS_r[0])
        self.ypoints.append(T_WS_r[1])
        self.zpoints.append(T_WS_r[2])
        self.T_WS_r = T_WS_r
        self.T_WS_C = T_WS_C
        #self.locks['position'].release()
        #self.max_time_secs = instance.header.stamp.secs
        #self.max_time_nsecs = instance.header.stamp.nsecs
        #print("self ypoints are " + str(self.ypoints))
        if len(self.ypoints) > 2:
            image = self.visualise(depth,camera_model,h,w)
            print(image.shape)
            return imutils.resize(image,width=w,height=h)
        else:
            return np.zeros((h,w,3),dtype=np.uint8)

    def plot_axes(self,line_length):

        #X Axes
        length = math.sqrt(self.T_WS_C[0,0]**2 + self.T_WS_C[1,0]**2 + self.T_WS_C[2,0]**2)
        x_x,x_y = [self.xpoints[-1],self.xpoints[-1]+self.T_WS_C[0,0]*length/line_length],[self.ypoints[-1],self.ypoints[-1]+self.T_WS_C[1,0]*length/line_length]
        x_z = [self.zpoints[-1],self.zpoints[-1]+self.T_WS_C[2,0]*length/line_length]
        print("X points are : " + str([x_x,x_y,x_z]))
        self.ax.plot3D(x_x,x_y,x_z, 'b-',linewidth=2)

        #Y Axes
        length = math.sqrt(self.T_WS_C[0,2]**2 + self.T_WS_C[1,2]**2 + self.T_WS_C[2,2]**2)
        y_x,y_y = [self.xpoints[-1],self.xpoints[-1]+self.T_WS_C[0,1]*length/line_length],[self.ypoints[-1],self.ypoints[-1]+self.T_WS_C[1,1]*length/line_length]
        y_z = [self.zpoints[-1],self.zpoints[-1]+self.T_WS_C[2,1]*length/line_length]
        self.ax.plot3D(y_x,y_y,y_z, 'g-',linewidth=2)

        #Z Axes
        length = math.sqrt(self.T_WS_C[0,2]**2 + self.T_WS_C[1,2]**2 + self.T_WS_C[2,2]**2)
        z_x,z_y = [self.xpoints[-1],self.xpoints[-1]+self.T_WS_C[0,2]*length/line_length],[self.ypoints[-1],self.ypoints[-1]+self.T_WS_C[1,2]*length/line_length]
        z_z = [self.zpoints[-1],self.zpoints[-1]+self.T_WS_C[2,2]*length/line_length]
        self.ax.plot3D(z_x,z_y,z_z, 'r-',linewidth=2)

    def check_max(self,point):
        if point[0] > max_x:
            max_x = point[0]

    def get_points(self,depth,camera_model,h,w):
        self.pointcloud = []
        for row in range(depth.shape[0]):
            for col in range(depth.shape[1]):
                if depth[row,col] > 0:
                    point = np.zeros((4,1))
                    point[:,0] = [col+1,row+1,depth[row,col],1]
                    arr = np.concatenate((self.T_WS_C,self.T_WS_r[:,None]),axis = 1)
                    self.pointcloud.append(np.matmul(arr,point)*camera_model["depth_baseline"])
                    #print("The point location in 3d is : " + str(td_loc))
                    #, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs)

    def plot_points(self):
        for tdloc in self.pointcloud:
            self.ax.scatter(tdloc[0],tdloc[1],tdloc[2],s=0.1)


    def visualise(self,depth,camera_model,h,w):
        self.ax.cla()
        #self.locks['position'].acquire()

        self.ax.set_xlim([min(self.xpoints),max(self.xpoints)])
        self.ax.set_ylim([min(self.ypoints),max(self.ypoints)])
        self.ax.set_zlim([min(self.zpoints),max(self.zpoints)])
        #self.ax.set_xlim([-5,5])
        #self.ax.set_ylim([-5,5])
        #self.ax.set_zlim([-5,5])
        line1, = self.ax.plot3D(self.xpoints,self.ypoints,self.zpoints, 'k-')
        #self.get_points(depth,camera_model,h,w)
        #self.plot_points()
        self.plot_axes((max(self.xpoints)-min(self.xpoints))*0.05)
        #self.locks['position'].release()
        self.fig.canvas.draw()
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height),int(width),3)
        #print("Len points is " + str(len(self.pointcloud)))
        self.fig.canvas.flush_events()
        return image
