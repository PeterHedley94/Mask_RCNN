#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np, cv2
#RUNNING WITHOUT ROS
ROOT_DIR = os.path.abspath("./mask_rcnn")

class pose_visualiser:
    def __init__(self,w,h):
        self.no_points = 50000
        self.points = np.zeros((self.no_points,2),dtype = np.float64)
        self.scale = 1
        self.count = 0
        #min then max
        self.xlims = [-10**-100,10**-100]
        self.ylims = [-10**-100,10**-100]
        self.width = w
        self.height = h
        self.image = np.zeros((h,w,3))
        self.objects = np.zeros((2,0))

    def add_points(self,T_WS_r):
        if self.count < self.no_points:
            self.count += 1
        else:
            self.points[0:-1,:] = self.points[1:,:]
        self.points[self.count,0] = T_WS_r[0]
        self.points[self.count,1] = T_WS_r[1]

    def image_coords(self,points):
        points[:self.count,0] = (self.width/2) + (points[:self.count,0]*self.scale)/2
        points[:self.count,1] = (self.height/2) - (points[:self.count,1]*self.scale)/2
        return points[:self.count,:]

    def no_count_image_coords(self,points):
        points[:,0] = (self.width/2) + (points[:,0]*self.scale)/2
        points[:,1] = (self.height/2) - (points[:,1]*self.scale)/2
        return points


    def plot_bike_path(self,cycle_model):

        count = 1

        x,y,z,time_,uncert,val = cycle_model.predict()
        array = np.zeros((2,len(uncert)))
        if val == True:
            array[0,:],array[1,:] = x,y
            to_plot = self.no_count_image_coords(array.transpose())
            for point in range(to_plot.shape[0]):
                radius = uncert[point]
                point = (int(to_plot[point,0]),int(to_plot[point,1]))
                cv2.circle(self.image,point,int(radius*self.scale),(255,0,255),1)

    def plot(self,objects,cycle_model):
        self.image = np.zeros((self.height,self.width,3))

        objects = objects[:,objects[1,:]<self.points[-1,1]+50]
        objects = objects[:,objects[0,:]<self.points[-1,0]+50]
        objects = objects[:,objects[1,:]>self.points[-1,1]-50]
        objects = objects[:,objects[0,:]>self.points[-1,0]-50]

        self.xlims = [min(min(self.points[:,0]*0.5),self.xlims[0]),max(max(self.points[:,0]*1.5),self.xlims[1])]
        self.ylims = [min(min(self.points[:,1]*0.5),self.ylims[0]),max(max(self.points[:,1]*1.5),self.ylims[1])]

        self.objects = np.concatenate([objects,self.objects], axis = 1)

        if objects.shape[1] > 0:
            if self.xlims[0] > min(self.objects[0,:]):
                self.xlims[0] = min(self.objects[0,:])*0.5
            elif self.xlims[1] < max(self.objects[0,:]):
                self.xlims[1] = max(self.objects[0,:])*1.5

            if self.ylims[0] > min(self.objects[1,:]):
                self.ylims[0] = min(self.objects[1,:])*0.5
            elif self.ylims[1] < max(self.objects[1,:]):
                self.ylims[1] = max(self.objects[1,:])*1.5

        self.xlims[0] = min(self.xlims[0],-10**-3)
        self.xlims[1] = max(self.xlims[1],10**-3)
        self.ylims[0] = min(self.ylims[0],-10**-3)
        self.ylims[1] = max(self.ylims[1],10**-3)
        xscale = self.width/(self.xlims[1] - self.xlims[0])
        yscale = self.height/(self.ylims[1] - self.ylims[0])
        if xscale < yscale:
            self.scale = xscale
        else:
            self.scale = yscale

        to_plot = self.image_coords(self.points.copy())
        if self.count > 2:
            cv2.polylines(self.image,np.int32([to_plot]),False,(255,255,255),1)

        self.plot_bike_path(cycle_model)
        return self.image
