#!/usr/bin/env python
import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
from image_class import *
import cv2
import random
from mykalman import *
"""
   Tracking of rotating box.
"""
# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    long = int

import cv2 as cv
from math import cos, sin, sqrt
import numpy as np

count = 0

if __name__ == "__main__":

    def draw_rectangle(img,centre_x,centre_y,width,height, color):
        pt1 = (int(centre_x-width/2),int(centre_y-height/2))
        pt2 = (int(centre_x+width/2),int(centre_y+height/2))
        cv2.rectangle(img,pt1,pt2,color)

    def calc_point(angle):
        return (np.around(img_width/2 + img_width/3*cos(angle), 0).astype(int),
                np.around(img_height/2 - img_width/3*sin(angle), 1).astype(int))

    img_height = 500
    img_width = 500

    code = long(-1)

    cv.namedWindow("Kalman")

    while True:
        start_angle = 0.5
        state = np.array([100,100,50,50,0,0,0,0],dtype = np.float64)
        state = state[:,None]
        #print(state.shape)
        state[0:2,0] = calc_point(start_angle)
        #print(state)
        v1,v2,v3,v4,v5,v6,v7,v8,v9,v11,v12,v13,v14 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 1, 8, 8, 1]#[1, 8, 8, 1, 1, 8, 8, 1, 8, 1, 1, 8, 8]#r_()
        #if all( i==8 for i in [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14]) :
            #break
        #v1 = 1 # LESS SURE COULD BE 1
        #v2,v3,v6,v7,v14 = [8, 8, 8,8, 8]#[1, 8, 8, 1, 1, 8, 8, 8, 8, 4, 1, 1, 8, 8]
        print("Values are : " + str([v1,v2,v3,v4,v5,v6,v7,v8,v9,v11,v12,v13,v14]))
        #kalman = cv.KalmanFilter(8,4, 0)
        kalman = my_kalman(8,4)

        kalman.F = np.eye(8)
        deltat = 1.0
        kalman.F[:4,4:] = np.diag(np.array([deltat]*4,dtype = np.float64))
        #Hk ?
        kalman.H = np.zeros((4,8))
        kalman.H[:4,:4] = np.eye(4)
        #Q?
        kalman.Q =  np.diag(np.array([5,5,5,5,10,10,10,10],dtype = np.float64)) # 1e-5 *
        #R?
        kalman.R = np.diag(np.array([5,5,5,5],dtype = np.float64)) # 1e-1 *
        # 8x8
        kalman.errorCovPost = np.eye(8)# 1.
        #Is this right?? K?
        kalman.statePost = state.copy()

        '''
        #Fk
        kalman.transitionMatrix = np.eye(8)
        deltat = 1.0
        kalman.transitionMatrix[:4,4:] = np.diag(np.array([deltat]*4,dtype = np.float64))
        #Hk ?
        kalman.measurementMatrix = np.zeros((4,8))
        kalman.measurementMatrix[:4,:4] = np.eye(4)
        #Q?
        kalman.processNoiseCov =  np.diag(np.array([5,5,5,5,10,10,10,10],dtype = np.float64)) # 1e-5 *
        #R?
        kalman.measurementNoiseCov = np.diag(np.array([5,5,5,5],dtype = np.float64)) # 1e-1 *
        # 8x8
        kalman.errorCovPost = np.eye(8)# 1.
        #Is this right?? K?
        kalman.statePost = state.copy()'''

        count = 0
        while True:
            img = np.zeros((img_height, img_width, 3), np.uint8)

            prediction = kalman.predict()


            start_angle = start_angle + start_angle* 0.01
            state[4:6,0] = calc_point(start_angle)-state[0,0:2]
            state[0:2,0] = calc_point(start_angle)

            #measurement = kalman.measurementNoiseCov * np.random.randn(v11,v12)

            # generate measurement
            #measurement = np.dot(kalman.measurementMatrix, state) + measurement
            #print("Measurement shape is " + str(measurement.shape))
            measurement = state.copy()
            kalman.correct(measurement[:4,0])

            #process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(1, 8)
            #print("Process noise shape is " + str(process_noise.shape))
            #state = np.dot(kalman.transitionMatrix, state) #+ process_noise
            #print("State shape : " + str(state.shape))

            print("X pos is : " + str(state[0,0]) + " Predicted as : " + str(prediction[0]))
            print("Y pos is : " + str(state[1,0]) + " Predicted as : " + str(prediction[1]))
            print("Width is : " + str(state[2,0]) + " Predicted as : " + str(prediction[2]))
            print("Height is : " + str(state[3,0]) + " Predicted as : " + str(prediction[3]))
            print("X vel is : " + str(state[4,0]) + " Predicted as : " + str(prediction[4]))
            print("Y vel is : " + str(state[5,0]) + " Predicted as : " + str(prediction[5]))
            print("Width vel is : " + str(state[6,0]) + " Predicted as : " + str(prediction[6]))
            print("Height vel is : " + str(state[7,0]) + " Predicted as : " + str(prediction[7]))
            print("Kalman state pre is " + str(kalman.statePre))

            draw_rectangle(img,state[0,0],state[1,0],state[2,0],state[3,0],(0, 0, 255))
            draw_rectangle(img,prediction[0],prediction[1],prediction[2],prediction[3],(0, 255, 0))


            cv.imshow("Kalman", img)
            code = cv.waitKey(100)
            if code != -1:
                break


        if code in [27, ord('q'), ord('Q')]:
            break

    cv.destroyWindow("Kalman")
