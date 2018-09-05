#!/usr/bin/env python
import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import math, numpy as np

class collision_detector:
    def __init__(self):
        self.var = 0

    def predict_collision(self,cycle_model,mask_rcnn):
        x,y,z,time_step,uncertainty,val = cycle_model.predict()

        if val:
            for j in range(mask_rcnn.no_rois):
                mask_rcnn.object_collision[j] = False
                for i in range(1,len(x)):
                    x_c,y_c,c_uncert = x[i],y[i],uncertainty[i]
                    vals,o_uncert = mask_rcnn.kalman[j].predict_seconds(i*time_step)
                    #check the object is moving
                    if abs(np.mean(vals[5:8])) > 0.4:
                        o_x,o_y = vals[:2]
                        IOU = self.get_iou([o_x,o_y],[x[i],y[i]],o_uncert,c_uncert)
                        if IOU > 0.003:
                            mask_rcnn.object_collision[j] = True
        return mask_rcnn

    def get_iou(self,pt1,pt2,rad1,rad2):
        #check circles intersect
        d = self.get_distance(pt1,pt2)
        if d < (rad1 + rad2):
            a = rad1**2
            b = rad2**2

            #check circles are not within one another
            if d < abs(rad1-rad2):
                intersect = math.pi * min(a,b)
            #if not calc intersect area
            else:
                x = (a-b + d**2) / (2*d)
                z = x**2
                y = math.sqrt(a-z)
                intersect = a* math.asin(y/rad1) + b * math.asin(y/rad2) - y*(x + math.sqrt(z+b-a))
        else:
            intersect = 0

        IOU = intersect/(math.pi*(rad1**2) + math.pi*(rad2**2) - intersect)
        return IOU

    def get_distance(self,p1,p2):
        diff = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        return math.sqrt(diff)
