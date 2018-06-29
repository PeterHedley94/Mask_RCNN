import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
print(os.getcwd())
print(parentdir)
import synchronous_mrcnn as test
from utils.image_class import *
import numpy as np
import unittest
model_ = test.model(1000,800)



class ComponentTestCase(unittest.TestCase):

    def test_matching(self):


        array = np.array([[0.1,1,1],[1,1,0.1],[1,0.1,1]])
        indices = model_.max_index_selector(array)
        ans = [1,3,2]

        for val,an in zip(indices,ans):
            self.assertEqual(val, an)

        array = np.array([[0.1,1,1],[0.1,0.2,0.1],[0.1,0.2,0.3]])
        indices = model_.max_index_selector(array)
        ans = [1,3,2]

        for val,an in zip(indices,ans):
            self.assertEqual(val, an)







if __name__ == '__main__':
    unittest.main()
