import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
print(os.getcwd())
print(parentdir)
import mask_and_camshift as test
from mask_rcnn.image_class import *
import numpy as np
import unittest
model_ = test.model()


''''ry1, rx1, ry2, rx2'''
test_roi = [1,1,2,2]
test_list_roi = []
number_test = 2
for row in range(number_test):
    for col in range(number_test):
        test_list_roi.append([test_roi[0]+row,test_roi[1]+col,test_roi[2]+row,test_roi[3]+col])
test_list_roi = np.array(test_list_roi)



class ComponentTestCase(unittest.TestCase):

    def test_iou(self):
        self.assertEqual(model_.bb_intersection_over_union(test_list_roi[0],test_list_roi[0]), 1)
        self.assertEqual(model_.bb_intersection_over_union(test_list_roi[1],test_list_roi[0]), 0)


    def test_equal_rois(self):
        print("\nTEST EQUAL ROIS")
        res_ = roi_class(test_list_roi,12)
        [indices,newer] = model_.match_ROIs(res_, res_)
        for i1,i2 in zip(indices,np.arange(indices.shape[0])+1):
            self.assertEqual(i1,i2)

    def test_unequal_rois(self):
        print("\nTEST UNEQUAL ROIS")
        slice = number_test
        res_1 = roi_class(test_list_roi[:slice,:],12)
        res_2 = roi_class(test_list_roi[slice:,:],12)
        [indices,newer] = model_.match_ROIs(res_1, res_2)
        for i1,i2 in zip(indices,np.arange(indices.shape[0])+1):
            self.assertEqual(i1,i2+indices.shape[0])

    def test_no_roi(self):
        print("\nTEST NO ROI")
        res_1 = roi_class(np.empty((0,4)),12)
        res_2 = roi_class(test_list_roi,12)

        [indices,newer] = model_.match_ROIs(res_1, res_2)
        for i1,i2 in zip(indices,np.arange(indices.shape[0])+1):
            self.assertEqual(i1,i2)

        [indices,newer] = model_.match_ROIs(res_2, res_1)
        self.assertEqual(len(indices),0)
        for i in newer.lives:
            self.assertEqual(i,3)

    def test_roi_lives(self):
        print("\n TEST ROI LIVES")
        slice = number_test
        res_1 = roi_class(test_list_roi[:slice],12)
        res_2 = roi_class(test_list_roi[:slice-1],12)
        [indices,newer] = model_.match_ROIs(res_1, res_2)
        self.assertEqual(newer.lives[-1],3)


if __name__ == '__main__':
    unittest.main()
