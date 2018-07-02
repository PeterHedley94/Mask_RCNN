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



class ComponentTestCase(unittest.TestCase):

    def test_matching(self):

        old_ids = np.arange(3)
        print("Old ids are : " + str(old_ids))
        array = np.array([[0.9,0,0],[0,0,0.9],[0,0.9,0]])
        print("array is : " + str(array))

        [indices,not_in] = max_index_selector(array,old_ids,0.1)
        ans = [0,2,1]

        for val,an in zip(indices,ans):
            self.assertEqual(val, an)

        old_ids = np.arange(10)
        array = np.array([[0.9,0,0],[0.9,0.7,0.9],[0.9,0.7,0.3]])
        [indices,not_in] = max_index_selector(array,old_ids,0.1)
        ans = [0,2,1]

        for val,an in zip(indices,ans):
            self.assertEqual(val, an)
        ind_ans = [3,4,5,6,7,8,9]

        print(not_in)

        for ind,an in zip(not_in,ind_ans):
            self.assertEqual(ind, an)

        old_ids = np.arange(10)
        array = np.array([[0.9,0,0],[0.9,0.7,0.9],[0.9,0.3,0.3]])
        [indices,not_in] = max_index_selector(array,old_ids,0.5)
        ans = [0,2,-1]

        print(not_in)


        for ind,an in zip(indices,ans):
            self.assertEqual(ind, an)

        ind_ans = [1,3,4,5,6,7,8,9]

        for ind,an in zip(not_in,ind_ans):
            self.assertEqual(ind, an)









if __name__ == '__main__':
    unittest.main()
