import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
import cv2
from matplotlib import pyplot as plt
import shutil
import numpy as np
import cv2
from matplotlib import pyplot as plt


class BRISK_class:
    def __init__(self):
        # Initiate BRISK detector
        self.brisk = cv2.BRISK_create()
        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_des_key(self,img,rois):
        des = []
        key = []

        for roi in rois:
            key1,des1 = self.brisk.detectAndCompute(img[roi[0]:roi[2],roi[1]:roi[3]],None)
            des.append(des1)
            '''
            if des1 is  None:
                des.append([])
            else:
                des.append(des1)
            '''
            key.append(key1)

        return des,key

    def get_matches(self,des1,des2):
        if des1 is not None and des2 is not None:
            print("DOES THIS !!!!!!!!!!!!")
            return self.bf.match(des1,des2)
        else:
            return []

    def get_probabilities(self,array):
        #1500 is an arbitrary value deemed to be very little resemblence between objects
        array = (1500-array)/1500
        array[array<=0] = 10**-3
        #This method is correct around 0.75 of the time therefore weight accordingly
        return array*0.75


    def get_sum_matches(self,older,old_,newer,new_):
        sum = 0
        print("des length is : " + str(len(older.descriptors)))
        print("Rois shape is " + str(older.roi.shape))
        print("Old_ is : " + str(old_))
        print("des length is : " + str(len(newer.descriptors)))
        print("New is : " + str(new_))
        matches = self.get_matches(older.descriptors[old_],newer.descriptors[new_])
        print("BRISK matches are : " + str(matches))
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        if len(matches) == 0:
            #1500 is an arbitrary value deemed to be very little resemblence between objects
            return 1500
        elif len(matches) < 10:
            for i in matches:
                sum += i.distance
        else:
            for i in matches[:10]:
                sum += i.distance
        return sum
