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
import cv2,time
import threading


def print_similarity_matrix(model):
    print("Buffer LENGTH IS " + str(len(model.buffers['mask'])))
    if(len(model.buffers['mask']) > 1):
        old_features = model.buffers['mask'][0].features
        old_id= model.buffers['mask'][0].id

        new_features = model.buffers['mask'][-1].features
        new_id= model.buffers['mask'][-1].id

        similarity = np.zeros((max(new_id)+1,max(old_id)+1))

        print("Old ids are : " + str(old_id))

        for old_roi in range(len(old_id)):
            for new_roi in range(len(new_id)):
                #print(new_features[new_roi,:,:,:]-old_features[old_roi,:,:,:])
                similarity[new_id[new_roi],old_id[old_roi]] = np.absolute(np.mean(new_features[new_roi,:,:,:]-old_features[old_roi,:,:,:]))
        print(similarity)

        print("New ids are : " + str(new_id))
        print(np.argmin(similarity[1:,1:],axis = 0))




print(os.getcwd())

path = '/home/peter/catkin_ws/src/mask_rcnn/src/mask_rcnn/gold.avi'
cap = cv2.VideoCapture(path)
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ic = test.model(frame_width,frame_height)

print(os.getcwd())
frame_no = 0

def print_progress(counter,total):
    inc = 5 #print every 10%
    percentage = str(counter/total * 100)
    title_string = percentage + str('% [')
    for i in range(round(counter/total * 100/inc)):
        title_string = title_string + '=='
    title_string = title_string + '>'
    for i in range(round(100/inc-round(counter/total * 100/inc))):
        title_string = title_string + '__'
    title_string = title_string + ']'
    print(title_string)

if(cap.isOpened() == False):
  print("Failed to get camera")
else:
    length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Found " + str(length_video) + " images")
    #rospy.init_node('mask_rcnn', anonymous=True)
    #rospy.spin()
    print("Located camera")
    # take first frame of the video
    while(1):
        ret,frame = cap.read()
        print_progress(frame_no,length_video)
        frame_no += 1
        if ret ==True and frame_no > 895:
            ic.callback(frame)
            print_similarity_matrix(ic)
            cv2.imshow("frame", frame)
            cv2.waitKey(10)
