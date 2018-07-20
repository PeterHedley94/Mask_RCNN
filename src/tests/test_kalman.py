import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import cv2
print(os.getcwd())
print(parentdir)
from Tracking.Tracker import *
from utils.image_class import *
import numpy as np
import unittest

def mm(size):
    return np.zeros((1080, 1920, size))

def mf(size):
    return np.zeros((size, 14, 14, 256))

def cl(size):
    return [255,255,255]*size

nt = 2

def get_rois(start = 1):
    test_roi = [start,start,start+1,start+1]
    test_list_roi = []
    for row in range(nt):
        for col in range(nt):
            test_list_roi.append([test_roi[0]+row,test_roi[1]+col,test_roi[2]+row,test_roi[3]+col])
    return(np.array(test_list_roi))

def get_noisy_rois(start = 1):
    test_roi = [start,start,start+1,start+1]
    test_list_roi = []
    for row in range(nt):
        for col in range(nt):
            noise = np.random.normal(0, 10**-5, 4)
            test_list_roi.append([test_roi[0]+row+noise[0],test_roi[1]+col+noise[1],
            test_roi[2]+row+noise[2],test_roi[3]+col+noise[3]])
    return(np.array(test_list_roi))
'''
def get_roi_history(length = 10,noise=False):
    if noise:
        res = roi_class(get_noisy_rois(length),12,np.arange(nt),cl(nt),mm(nt),mf(nt),10,10)
    else:
        res = roi_class(get_rois(length),12,np.arange(nt),cl(nt),mm(nt),mf(nt),10,10)
    arr = (np.arange(10)+1.5)[::-1]
    res.centre_x[:length] = np.transpose(np.array([arr,arr+1,arr+2,arr+3]))
    res.centre_y[:length] = np.transpose(np.array([arr,arr+1,arr+2,arr+3]))
    res.roi_width[:length] = np.ones((10,4))
    res.roi_height[:length] = np.ones((10,4))
    return res
'''
class ComponentTestCase(unittest.TestCase):

    def test_prediction_sequence(self):
        older = roi_class(get_noisy_rois(1),12,np.arange(nt),cl(nt),mm(nt),mf(nt),10,10)
        for i in range(1,20):
            temp = roi_class(get_noisy_rois(i),12,np.arange(nt),cl(nt),mm(nt),mf(nt),10,10)
            for roi in range(nt):
                older.kalman[roi].predict()
                measurement = np.array([temp.centre_x[roi],temp.centre_y[roi],temp.roi_width[roi],temp.roi_height[roi]])
                print("Measurement shape is " + str(measurement.shape))
                older.kalman[roi].correct(measurement)

        newer = roi_class(get_noisy_rois(20),12,np.arange(nt),cl(nt),mm(nt),mf(nt),10,10)
        for old_ in range(nt):
            older.kalman[old_].predict()
            for new_ in range(nt):
                print(Kalman_predict(older,old_,newer,new_))
                print("                                ")






if __name__ == '__main__':
    unittest.main()
