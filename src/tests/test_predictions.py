import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
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

''''ry1, rx1, ry2, rx2'''

nt = 2

def get_rois(start = 1):
    test_roi = [start,start,start+1,start+1]
    test_list_roi = []
    for row in range(nt):
        for col in range(nt):
            test_list_roi.append([test_roi[0]+row,test_roi[1]+col,test_roi[2]+row,test_roi[3]+col])
    return(np.array(test_list_roi))

def get_roi_history(length = 10):
    res = roi_class(get_rois(length),12,np.arange(nt),cl(nt),mm(nt),mf(nt),10,10)
    arr = (np.arange(10)+1.5)[::-1]
    res.centre_x[:length] = np.transpose(np.array([arr,arr+1,arr+2,arr+3]))
    res.centre_y[:length] = np.transpose(np.array([arr,arr+1,arr+2,arr+3]))
    res.roi_width[:length] = np.ones((10,4))
    res.roi_height[:length] = np.ones((10,4))
    return res

class ComponentTestCase(unittest.TestCase):
    def test_gradient(self):
        res_1 = get_roi_history(10)
        [grad,gradunc] = get_gradient(res_1.centre_x[:10,0])
        print("RES centre" + str(res_1.centre_x[:10,0]))
        self.assertEqual(0,gradunc)
        for val,an in zip(grad,[1]*len(grad)):
            self.assertEqual(val, an)

    def test_prediction(self):
        res_1 = get_roi_history(10)
        res_2 = roi_class(get_rois(11),12,np.arange(nt),cl(nt),mm(nt),mf(nt),10,10)

        print(predict_ROI(res_1,0,res_2,0))




if __name__ == '__main__':
    unittest.main()
