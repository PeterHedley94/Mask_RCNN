import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
from utils.common import *
import numpy as np, cv2, shutil,math
from matplotlib import pyplot as plt
IMAGE_COUNT = 0

class BRISK_class:
    def __init__(self):
        # Initiate BRISK detector
        self.brisk = cv2.BRISK_create()
        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    #Get keypoints and descriptors of a mask
    def get_des_key(self,img,rois,masks):
        global IMAGE_COUNT
        des,key = [],[]

        for roi,mask in zip(rois,range(masks.shape[2])):
            zorro = img.copy()
            '''
            #For 3 colour channels only use masked area
            for i in range(3):
                np.multiply(img[:,:,i],masks[:,:,mask],out = zorro[:,:,i])
            '''
            '''
            img_cpy = zorro[roi[0]:roi[2],roi[1]:roi[3]].copy()
            gray = cv2.cvtColor(img_cpy,cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray,2,3,0.04)
            #result is dilated for marking the corners, not important
            dst = cv2.dilate(dst,None)
            # Threshold for an optimal value, it may vary depending on the image.
            img_cpy[dst>0.01*dst.max()]=[0,0,255]
            image_name = os.path.join("cornerHarris",str(IMAGE_COUNT) + ".jpg")
            cv2.imwrite(image_name,img_cpy)
            IMAGE_COUNT += 1
            key1,des1 = self.brisk.compute(zorro[roi[0]:roi[2],roi[1]:roi[3]],dst)
            '''
            key1,des1 = self.brisk.detectAndCompute(zorro[roi[0]:roi[2],roi[1]:roi[3]],None)
            des.append(des1)
            key.append(key1)
        return des,key

    #Return all matches from BRISK
    def get_matches(self,des1,des2):
        if des1 is not None and des2 is not None:
            return self.bf.match(des1,des2)
        else:
            return []

    #Weight to make less prominant
    def get_probabilities(self,array):
        return array*0.75

    #Count number of good matches and divide by number of keypoint in ROIs
    def get_match_score(self,older,old_,newer,new_):
        global BRISK_THRESHOLD,BRISK_MIN_VALUE
        no = 0
        matches = self.get_matches(older.descriptors[old_],newer.descriptors[new_])
        if len(matches) == 0:
            return 0.1
        else:
            for i in matches:
                if i.distance < BRISK_THRESHOLD:
                    no += 1
        no_key = len(older.keypoints[old_]) + len(newer.keypoints[new_])
        #print(" No is : " + str(no) + " total is : " + str(no_key))
        if no == 0:
            return BRISK_MIN_VALUE
        if no_key > 0:
            return no**2/no_key * math.log(no_key,2)
        else:
            return 0

    '''

    def draw_rectangle(self,img,roi, colour):
        cv2.rectangle(img,(roi[1],roi[0]),(roi[3],roi[2]),colour,3)

    def get_probabilities(self,array):
        #1500 is an arbitrary value deemed to be very little resemblence between objects
        array = (1500-array)/1500
        array[array<=0] = 10**-3
        #This method is correct around 0.75 of the time therefore weight accordingly
        return array*0.75'''
    '''
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
    '''
