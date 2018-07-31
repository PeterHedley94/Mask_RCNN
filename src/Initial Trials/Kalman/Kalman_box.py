#!/usr/bin/env python
import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
from image_class import *
import cv2
import random
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

    class counter:
        def __init__(self):
            self.values = [1,8]
            self.index = 0

        def increment(self):
            if self.index == len(self.values)-1:
                self.index = 0
                return True
            else:
                self.index += 1
                return False

    values = []
    for i in range(13):
        values.append(counter())

    def calc_point(angle):
        return (np.around(img_width/2 + img_width/3*cos(angle), 0).astype(int),
                np.around(img_height/2 - img_width/3*sin(angle), 1).astype(int))

    def r():
        global count
        r_ = random.randint(0,8)
        return r_

    def r_():
        global values
        array = [2,4,8]
        val_index = len(values)
        while(val_index >= 0 or values[val_index].increment() == True):
            val_index -= 1
        r = []
        for i in values:
            r.append(i.values[i.index])
        return r

    img_height = 500
    img_width = 500
    start_angle = 0.5
    code = long(-1)


    #print(values[val_index].increment())
    '''
    for i in range(10):
        v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14 = r_()
        with open("hello.txt",'a') as file:
            file.write("Values are : " + str([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14]) + "\n")
        print("Values are : " + str([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14]))'''

    cv.namedWindow("Kalman")

    while True:
        try:
            state = np.array([100,100,50,50,0,0,0,0],dtype = np.float64)
            state = state[:,None]
            #print(state.shape)
            state[0:2,0] = calc_point(start_angle)
            #print(state)
            v1,v2,v3,v4,v5,v6,v7,v8,v9,v11,v12,v13,v14 = r_()
            v10 = -300
            #if all( i==8 for i in [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14]) :
                #break
            #v1 = 1 # LESS SURE COULD BE 1
            #v2,v3,v6,v7,v14 = [8, 8, 8,8, 8]#[1, 8, 8, 1, 1, 8, 8, 8, 8, 4, 1, 1, 8, 8]
            print("Values are : " + str([v1,v2,v3,v4,v5,v6,v7,v8,v9,v11,v12,v13,v14]))
            kalman = cv.KalmanFilter(v8,v9, 0)
            #Fk
            kalman.transitionMatrix = np.eye(8)
            deltat = 1.0
            kalman.transitionMatrix[:4,4:] = np.diag(np.array([deltat]*4,dtype = np.float64))
            #print(kalman.transitionMatrix)

            #Hk ?
            kalman.measurementMatrix = 1. * np.ones((v1, v2))

            #Pk?
            kalman.processNoiseCov = 1e-5 * np.eye(v3)

            #Rk?
            kalman.measurementNoiseCov = 1e-1 * np.ones((v4, v5))

            #Qk?
            kalman.errorCovPost = 1. * np.ones((v6, v7))

            #Is this right?? K?
            kalman.statePost = state
            count = 0
            while True:
                img = np.zeros((img_height, img_width, 3), np.uint8)

                prediction = kalman.predict()

                measurement = kalman.measurementNoiseCov * np.random.randn(v11,v12)

                # generate measurement
                measurement = np.dot(kalman.measurementMatrix, state) + measurement
                #print("Measurement shape is " + str(measurement.shape))
                kalman.correct(state)

                process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(1, 8)
                #print("Process noise shape is " + str(process_noise.shape))
                state = np.dot(kalman.transitionMatrix, state) #+ process_noise
                #print("State shape : " + str(state.shape))

                start_angle = start_angle + start_angle* 0.01
                state[4:6,0] = calc_point(start_angle)-state[0,0:2]
                state[0:2,0] = calc_point(start_angle)

                draw_rectangle(img,state[0,0],state[1,0],state[2,0],state[3,0],(0, 0, 255))
                #draw_rectangle(img,prediction[0],prediction[1],prediction[2],prediction[3],(0, 255, 0))


                cv.imshow("Kalman", img)
                code = cv.waitKey(100)
                if code != -1:
                    break
                count += 1
                if count == 20:
                    print("CORRECT ###############################################################")
                    print("Values are : " + str([v1,v2,v3,v4,v5,v6,v7,v8,v9,v11,v12,v13,v14]))
                    with open("hello.txt",'a') as file:
                        file.write("Values are : " + str([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14]) + "\n")
        except:
            continue

        if code in [27, ord('q'), ord('Q')]:
            break

    cv.destroyWindow("Kalman")
