#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np, cv2
#RUNNING WITHOUT ROS
ROOT_DIR = os.path.abspath("./mask_rcnn")

class obj_map_visualiser:
    def __init__(self,w,h):
        self.width = w
        self.height = h
        self.image = np.zeros((h,w,3))
        self.xlims = [0,0]
        self.ylims = [0,0]
        self.xpoints = []
        self.ypoints = []
        self.fontScale = 0.2
        self.fontColor=(255,255,255)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def set_limits(self,xl,yl):
        self.xlims = xl
        self.ylims = yl

    def plot_object_path(self,img,m_,index):
        array = np.zeros((2,5))
        for i in range(0,5,1):
            x,y,z = m_.object_models[index].predict(float(i+1))
            array[:,i] = x,y
        to_plot = self.array_image_coords(array.transpose())
        #print("The points to plot are : " + str(to_plot))
        cv2.polylines(img,np.int32([to_plot]),False,(255,0,255),1)
        return img

    def x_y_from_cx_cy(array):
        centre_x,centre_y,width,height = array
        pt1 = (int(centre_x-width/2),int(centre_y-height/2))
        pt2 = (int(centre_x+width/2),int(centre_y+height/2))
        return pt1,pt2

    def array_image_coords(self,points):
        points[:,0] = (self.width/2) + (points[:,0]*self.scale)/2
        points[:,1] = (self.height/2) - (points[:,1]*self.scale)/2
        return points

    def image_coords(self,x,y):
        #print("Plotting Object at : " + str(x) + " , " + str(y))
        x = (self.width/2) + (x*self.scale)/2
        y = (self.height/2) - (y*self.scale)/2
        #print("Plotting Object on image at : " + str(x) + " , " + str(y))
        return (int(x),int(y))

    def plot(self,img,mrcnn_output,class_names):
        xscale = self.width/(self.xlims[1] - self.xlims[0])
        yscale = self.height/(self.ylims[1] - self.ylims[0])

        for x,y in zip(self.xpoints,self.ypoints):
            cv2.circle(img,self.image_coords(x,y),0,(100,0,0),3)
            
        if xscale < yscale:
            self.scale = xscale
        else:
            self.scale = yscale

        for i in range(mrcnn_output.no_rois):
            #self.image_coords(rois[i,0],rois[i,1])
            x,y = mrcnn_output.roi_dims_w[0,i],mrcnn_output.roi_dims_w[1,i]

            self.xpoints.append(x)
            self.ypoints.append(y)

            #Actual point
            cv2.circle(img,self.image_coords(x,y),0,(255,255,255),3)
            text = str(mrcnn_output.id[i]) + " : " + class_names[mrcnn_output.class_[i]]
            pos = self.image_coords(x+(10/self.scale)*2,y)
            cv2.putText(img,str(text), pos, self.font,self.fontScale, (255,255,255), thickness = 1)

            #Kalman Predicted Location
            x,y = mrcnn_output.kalman[i].statePre[[0,1]]
            cv2.circle(img,self.image_coords(x,y),0,(0,255,0),3)
            pos = self.image_coords(x+(10/self.scale)*2,y-(10/self.scale)*2)
            cv2.putText(img,str(text), pos, self.font,self.fontScale, (0,255,0), thickness = 1)

            #Predicted Path
            img = self.plot_object_path(img,mrcnn_output,i)



        #cv2.point(self.image,np.int32([to_plot]),False,(255,255,255),1)
        #cv2.imshow('m',self.image)
        #cv2.waitKey(2)
        return img
