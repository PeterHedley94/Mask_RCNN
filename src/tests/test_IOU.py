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
test_roi = [1,1,2,2]
test_list_roi = []
nt = 2
for row in range(nt):
    for col in range(nt):
        test_list_roi.append([test_roi[0]+row,test_roi[1]+col,test_roi[2]+row,test_roi[3]+col])
test_list_roi = np.array(test_list_roi)



class ComponentTestCase(unittest.TestCase):

    def test_iou(self):
        self.assertEqual(bb_intersection_over_union(test_list_roi[0],test_list_roi[0]), 1)
        self.assertEqual(bb_intersection_over_union(test_list_roi[1],test_list_roi[0]), 0)


    def test_equal_rois(self):
        print("\nTEST EQUAL ROIS")
        res_ = roi_class(test_list_roi,12,np.arange(nt*2),cl(nt*2),mm(nt*2),mf(nt*2))
        [indices,newer] = match_ROIs(res_, res_,0.75)

        for i1,i2 in zip(indices,np.arange(indices.shape[0])):
            self.assertEqual(i1,i2)

    def test_unequal_rois(self):
        print("\nTEST UNEQUAL ROIS")
        slice = nt
        res_1 = roi_class(test_list_roi[:slice,:],12,np.arange(nt),cl(nt),mm(nt),mf(nt))
        res_2 = roi_class(test_list_roi[slice:,:],12,np.arange((nt)),cl((nt)),mm((nt)),mf((nt)))
        [indices,newer] = match_ROIs(res_1, res_2,0.75)

        for i1,i2 in zip(indices,[2,3,0,1]):
            self.assertEqual(i1,i2)

    def test_no_roi(self):
        print("\nTEST NO ROI")
        res_1 = roi_class(np.empty((0,4)),12,np.arange(0),cl(0),mm(0),mf(0))
        res_2 = roi_class(test_list_roi,12,np.arange(nt*2),cl(nt*2),mm(nt*2),mf(nt*2))

        [indices,newer] = match_ROIs(res_1, res_2,0.75)

        for i1,i2 in zip(indices,np.arange(indices.shape[0])):
            self.assertEqual(i1,i2)

        [indices,newer] = match_ROIs(res_2, res_1, 0.75)
        for i1,i2 in zip(indices,np.arange(indices.shape[0])):
            self.assertEqual(i1,i2)

    def test_roi_lives(self):
        print("\n TEST ROI LIVES")
        slice = nt
        res_1 = roi_class(test_list_roi[:slice],12,np.arange(nt),cl(nt),mm(nt),mf(nt))
        res_2 = roi_class(test_list_roi[:slice-1],12,np.arange(nt),cl(nt),mm(nt),mf(nt))
        [indices,newer] = match_ROIs(res_1, res_2,0.75)
        self.assertEqual(newer.lives[-1],2)

        [indices,newer] = match_ROIs(res_1, res_1,0.75)
        for life in newer.lives:
            self.assertEqual(life,4)

    def test_remove_objects(self):
        print("\n TEST REMOVE OBJECTS")
        slice = nt
        res_1 = roi_class(np.empty((0,4)),12,np.arange(0),cl(0),mm(0),mf(0))

        res_2 = roi_class(test_list_roi,12,np.arange(nt*2),cl(nt*2),mm(nt*2),mf(nt*2))
        res_2.lives = [0] * len(res_2.lives)
        [indices,newer] = match_ROIs(res_2, res_1,0.75)
        self.assertEqual(len(newer.lives),0)
        self.assertEqual(len(indices),0)


if __name__ == '__main__':
    unittest.main()
