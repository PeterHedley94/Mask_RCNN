import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
print(os.getcwd())
print(parentdir)
import mask_and_camshift as test

#import unittest


def test_roi():
    'ry1, rx1, ry2, rx2'
    test_roi = [1,1,2,2]
    test_list_roi = test_roi
    for row in 100:
        for col in 100:
            test_list_roi.append([test_roi+row,test_roi+col,test_roi+row,test_roi+col)
    test_list_roi = np.array(test_list_roi)
    print(test_list_roi.shape)
    model_ = test.model()


#class ComponentTestCase(unittest.TestCase):




if __name__ == '__main__':
    test_roi()
