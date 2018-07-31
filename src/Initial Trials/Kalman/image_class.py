import collections
import numpy as np

class image_class:
    def __init__(self,frame_,time_):
        self.time = time_
        self.frame = frame_

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
    def __init__(self,roi_,time_,class_,colours,mask,features,des,key):
        self.time = time_
        self.roi = roi_
        self.hist = [None] * roi_.shape[0]
        self.id = np.arange(roi_.shape[0])
        self.colours = colours
        self.masks = mask
        self.features = features
        self.lives = [3] * roi_.shape[0]
        self.class_ = class_
        self.descriptors = des
        self.keypoints = key
        self.centre_x = np.zeros((10,roi_.shape[0]))
        self.centre_y = np.zeros((10,roi_.shape[0]))
        self.roi_width = np.zeros((10,roi_.shape[0]))
        self.roi_height = np.zeros((10,roi_.shape[0]))
        self.centre_x[:] = np.nan
        self.centre_y[:] = np.nan
        self.roi_width[:] = np.nan# = np.zeros((10,roi_.shape[0]))
        self.roi_height[:] = np.nan
        self.get_dimensions()

    def get_dimensions(self):
        self.centre_x = np.concatenate([(self.roi[None,:,0]+self.roi[None,:,2])/2,self.centre_x[:9,:]],0)
        self.centre_y = np.concatenate([(self.roi[None,:,1] + self.roi[None,:,3])/2,self.centre_y[:9,:]],0)
        self.roi_width = np.concatenate([self.roi[None,:,2] - self.roi[None,:,0],self.roi_width[:9,:]],0)
        self.roi_height = np.concatenate([self.roi[None,:,3] - self.roi[None,:,1],self.roi_height[:9,:]],0)

    def predict_next(self,roi_index):
        self.centre_x[1:10,roi_index] = self.centre_x[:9,roi_index]
        self.centre_y[1:10,roi_index] = self.centre_y[:9,roi_index]
        self.roi_width[1:10,roi_index] = self.roi_width[:9,roi_index]
        self.roi_height[1:10,roi_index] = self.roi_height[:9,roi_index]

        if np.isnan(self.centre_x[2,roi_index]) == False:
             self.centre_x[0,roi_index] = self.centre_x[1,roi_index] + (self.centre_x[1,roi_index]-self.centre_x[2,roi_index])
             self.centre_y[0,roi_index] = 2*self.centre_y[1,roi_index] - self.centre_y[2,roi_index]
             self.roi_width[0,roi_index] = 2*self.roi_width[1,roi_index] - self.roi_width[2,roi_index]
             self.roi_height[0,roi_index] = 2*self.roi_height[1,roi_index] - self.roi_height[2,roi_index]
        else:
             self.centre_x[0,roi_index] = self.centre_x[1,roi_index]
             self.centre_y[0,roi_index] = self.centre_y[1,roi_index]
             self.roi_width[0,roi_index] = self.roi_width[1,roi_index]
             self.roi_height[0,roi_index] = self.roi_height[1,roi_index]
