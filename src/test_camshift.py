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
import cv2,time
import threading
model_ = test.model()

''''ry1, rx1, ry2, rx2'''
test_roi = [1,1,2,2]
test_list_roi = []
number_test = 2
for row in range(number_test):
    for col in range(number_test):
        test_list_roi.append([test_roi[0]+row,test_roi[1]+col,test_roi[2]+row,test_roi[3]+col])
test_list_roi = np.array(test_list_roi)

print(os.getcwd())

path = '/home/peter/catkin_ws/src/mask_rcnn/src/mask_rcnn/vid.mp4'
cap = cv2.VideoCapture(path)
count = 0


ret,frame = cap.read()

while(ret == True):
    img = frame
    count += 1
    #path = '/home/peter/catkin_ws/src/mask_rcnn/src/tests/cyclists.jpg'
    #img = cv2.imread(path)
    #cv2.imshow('frame1',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    image = image_class(img,12)
    model_.buffers["image"].append(image)

    if (count % 10 == 0 or count == 1):
        mask_rcnn_thread = threading.Thread(target=model_.mask_predict)
        mask_rcnn_thread.daemon = True
        mask_rcnn_thread.start()
        model_.stop = True
        mask_rcnn_thread.join()

        mask_roi = model_.buffers['mask'][0].roi
        #model_.draw_rects(img,mask_roi)
        mask = model_.buffers["mask_image"][-1]
        mask = model_.draw_rects(mask,mask_roi)
        cv2.imshow('frame2',mask)
        print(" done mask")

    model_.stop = False
    camshift_thread = threading.Thread(target=model_.camshift)
    camshift_thread.daemon = True
    camshift_thread.start()
    model_.stop = True
    camshift_thread.join()
    img2 = model_.draw_rects(img.copy(),model_.buffers['camshift'][-1].roi)
    cv2.imshow('frame3',img2)
    camshift_roi = model_.buffers['camshift'][-1].roi
    ret,frame = cap.read()
    cv2.waitKey(0)

print("Failed")
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
path = '/home/peter/catkin_ws/src/mask_rcnn/src/tests/cyclists.jpg'

img = cv2.imread(path)
cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


image = image_class(img,12)
model_.buffers["image"].append(image)

mask_rcnn_thread = threading.Thread(target=model_.mask_predict)
mask_rcnn_thread.daemon = True
mask_rcnn_thread.start()
model_.stop = True
mask_rcnn_thread.join()

mask_roi = model_.buffers['mask'][0].roi
#model_.draw_rects(img,mask_roi)
mask = model_.buffers["mask_image"][-1]
mask = model_.draw_rects(mask,mask_roi)
cv2.imshow('frame',mask)

camshift_thread = threading.Thread(target=model_.camshift)
camshift_thread.daemon = True
camshift_thread.start()
model_.stop = True
camshift_thread.join()

img2 = model_.draw_rects(img.copy(),model_.buffers['camshift'][-1].roi)

cv2.imshow('frame2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

while(len(model_.buffers['camshift'][-1].roi) < 2):
    time.sleep(1)
camshift_roi = model_.buffers['camshift'][-1].roi

print("Mask :" + str(mask_roi))
print("CamShift" + str(camshift_roi))
'''
