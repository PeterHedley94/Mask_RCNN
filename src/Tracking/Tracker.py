#!/usr/bin/env python
import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
print("CWD : " + str(os.getcwd()))
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
print("CWD : " + str(os.getcwd()))
import numpy as np
from Tracking.BRISK import *
import math
#import camshift
from scipy.stats import multivariate_normal


def bb_intersection_over_union_probabilities(array):
    return array * 0.9

def bb_intersection_over_union(older,old_,newer,new_):
    boxA = older.roi[old_]
    boxB = newer.roi[new_]

    ff=0
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + ff) * (yB - yA + ff)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + ff) * (boxA[3] - boxA[1] + ff)
    boxBArea = (boxB[2] - boxB[0] + ff) * (boxB[3] - boxB[1] + ff)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if(float(boxAArea + boxBArea - interArea) > 0):
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    # return the intersection over union value
    return iou

def get_gradient(vals):
    found_first_value = False
    no_vals = np.count_nonzero(~np.isnan(vals))

    if(no_vals <= 1):
        x = np.zeros((10))
        x[:] = np.nan
        return [x,50]
    #*-1 as array is in reverse order (newest at pos 0)

    grad = np.gradient(vals[:no_vals])*-1
    #print("The gradient from values : " + str(np.gradient(vals[:no_vals])*-1) + " is " + str(grad))
    return [grad,np.std(grad)]

#Use equations of motion to predict new values
def predict_new_val(old_val,vel,vel_std,acc,acc_std,time =1):
    # x = x + ut + 1/2 *(at^2)
    #print("old_val,vel,vel_std,acc,acc_std")
    #print([old_val,vel,vel_std,acc,acc_std])
    pred = old_val + vel*time + (1/2 * acc * (time**2))
    pred_var = (vel_std**2)*time + 1/2 * (acc_std**2) * (time**2)
    #print("Variance is : " + str(pred_var))
    return pred,pred_var


def match_classes(older,old_,newer,new_):
    #print("Older classes shape is : " + str(older.class_[old_].shape))
    if older.class_[old_] == newer.class_[new_]:
        return 1
    else:
        return 0

def match_classes_probability(array):
    return array

def predict_ROI_probability(array):
    return array *0.9

def predict_ROI(older,old_,newer,new_):
    #print("Older centre x is : " + str(older.centre_x) )
    #print("########################################" + )
    if np.isnan(older.centre_x[1,old_]):
        [px_var,py_var,pw_var,ph_var] = [50,50,50,50]
        [px,py,pw,ph] = [older.centre_x[0,old_],older.centre_y[0,old_],older.roi_width[0,old_],older.roi_height[0,old_]]
    else:
        #Get velocity of change
        [vcx,vcx_std] = get_gradient(older.centre_x[:10,old_])
        [vcy,vcy_std] = get_gradient(older.centre_y[:10,old_])
        [vw,vw_std] = get_gradient(older.roi_width[:10,old_])
        [vh,vh_std] = get_gradient(older.roi_height[:10,old_])
        #Get acceleration of change
        [acx,acx_std] = get_gradient(vcx)
        [acy,acy_std] = get_gradient(vcy)
        [aw,aw_std] = get_gradient(vw)
        [ah,ah_std] = get_gradient(vh)
        #Get predicted new value
        #Maybe add weights
        [px,px_var] = predict_new_val(older.centre_x[0,old_],np.average(vcx),vcx_std,np.average(acx),acx_std)
        [py,py_var] = predict_new_val(older.centre_y[0,old_],np.average(vcy),vcy_std,np.average(acy),acy_std)
        [pw,pw_var] = predict_new_val(older.roi_width[0,old_],np.average(vw),vw_std,np.average(aw),aw_std)
        [ph,ph_var] = predict_new_val(older.roi_height[0,old_],np.average(vh),vh_std,np.average(ah),ah_std)

    cov_ = np.diag([px_var,py_var,pw_var,ph_var] + np.abs(np.random.normal(0, 10**-4, 4)))
    #print(cov_)
    #print([newer.centre_x[0,new_],newer.centre_y[0,new_],newer.roi_width[0,new_],newer.roi_height[0,new_]])
    #print([px,py,pw,ph])
    probability = multivariate_normal.pdf([newer.centre_x[0,new_],newer.centre_y[0,new_],newer.roi_width[0,new_],newer.roi_height[0,new_]],
    mean=[px,py,pw,ph], cov=cov_)
    #print("Probability is : " + str(probability))
    return probability

def match_ROIs(older, newer, BRISK_,IOU_threshold):

    #NO OBJECTS IN LAST FRAME
    if(older.roi.shape[0] == 0):
        print("No objects in last frame")
        return newer.id, newer

    #funcdict = {'bb_intersection_over_union': bb_intersection_over_union,
    funcdict = {'pred_roi':predict_ROI}
    #'bb_intersection_over_union': bb_intersection_over_union,'brisk':BRISK_.get_sum_matches,

    funcprob = {'pred_roi':predict_ROI_probability}
    #'bb_intersection_over_union': bb_intersection_over_union_probabilities,'brisk':BRISK_.get_probabilities}
    matches = np.zeros((len(funcdict.values()),newer.roi.shape[0],older.roi.shape[0]))
    classes = np.zeros((newer.roi.shape[0],older.roi.shape[0]))
    indices = np.zeros((newer.roi.shape[0]))

    for c,method_ in enumerate(funcdict.values()):
        for old_ in range(0,older.roi.shape[0]):
            for new_ in range(0,indices.shape[0]):
                matches[c,new_,old_] = method_(older,old_,newer,new_)
                if c == 0:
                    classes[new_,old_] = match_classes(older,old_,newer,new_)
    #print("Matches are : " + str(matches))

    for c,method_ in enumerate(funcprob.values()):
        matches[c,:,:] = method_(matches)
        matches[c,:,:] = np.multiply(matches[c,:,:],classes)
        if c > 0:
            matches[0,:,:] = matches[0,:,:] + matches[c,:,:]

    if newer.roi.shape[0] > 0:
        [final_indices,not_in] = max_index_selector(matches[0,:,:],0)
    else:
        final_indices = []
        not_in = older.id
    #print("Older ids are : " + str(older.id))
    if np.isnan(matches).any() == True:
        print("###################################################################################################################################################")
    #print("Matches are : " + str(matches))
    #print("final_indices are : " + str(final_indices))
    #print("not in is " + str(not_in))
    #print("newer_ids are : " + str(newer.id))
    #ADD OLDER ROIS IN THAT AREN'T IN THIS FRAME
    for old_ in not_in:
        if older.lives[old_]-1 > 0:
            #print("Does THis with : " + str(older.lives))
            newer.roi = np.concatenate((newer.roi, older.roi[None,old_,:]), axis=0)
            newer.id = np.concatenate((newer.id,older.id[old_,None]), axis=0)
            newer.class_ = np.concatenate((newer.class_,older.class_[old_,None]),axis = 0)
            #print("Newer Class: " + str(newer.class_.shape))
            #print("Older Class : " + str(older.centre_x[:,old_,None].shape))
            newer.centre_x = np.concatenate((newer.centre_x,older.centre_x[:,old_,None]),axis = 1)
            newer.centre_y =  np.concatenate((newer.centre_y,older.centre_y[:,old_,None]),axis = 1)
            newer.roi_width =  np.concatenate((newer.roi_width,older.roi_width[:,old_,None]),axis = 1)
            newer.roi_height = np.concatenate((newer.roi_height,older.roi_height[:,old_,None]),axis = 1)
            newer.predict_next(newer.roi_height.shape[1]-1)
            newer.lives.append(older.lives[old_]-1)
            newer.hist.append(older.hist[old_])
            newer.colours.append(older.colours[old_])
            newer.descriptors.append(older.descriptors[old_])
            newer.keypoints.append(older.keypoints[old_])

            if newer.masks.shape[2] == 0:
                newer.masks = older.masks[:,:,old_,None]
                newer.features = older.features[None,old_,:,:,:]
            else:
                newer.masks = np.concatenate([newer.masks,older.masks[:,:,old_,None]],axis = 2)
                newer.features = np.concatenate([newer.features,older.features[None,old_,:,:,:]],axis = 0)

    #Add objects that were in the older frame
    for c,index in enumerate(final_indices):
        if index < len(older.id):
            newer.id[c] = older.id[index]
            newer.colours[c] = older.colours[index]
            newer.centre_x[1:,c] = older.centre_x[0:9,index]
            newer.centre_y[1:,c] =  older.centre_y[0:9,index]
            newer.roi_width[1:,c] = older.roi_width[0:9,index]
            newer.roi_height[1:,c] = older.roi_height[0:9,index]

            if older.lives[index] < 7:
                newer.lives[c] = older.lives[index] +1

    #print("newer_ids are : " + str(newer.id))
    max_id = max(newer.id)+1

    for c,index in enumerate(final_indices):
        if index == -1:
            newer.id[c] = max_id
            max_id += 1
    if len(list(newer.id)) != len(set(list(newer.id))):
        print("###########################################################")
    #Add new objects that were not in older frame_
    return newer.id,newer


#CAN REMOVE MAX INDICES FROM THIS!!
def max_index_selector(array,threshold):
    already_used = []
    not_used = np.arange(array.shape[1])#np.array(old_ids)
    indices = [-1] * array.shape[0]
    max_indices = np.zeros(array.shape)
    values = np.where(array == array.max())

    #when found all new indices stop
    while(array.max() > threshold and len(already_used) < array.shape[0]):
        if values[1][0] not in already_used and indices[values[0][0]] == -1:
            already_used.append(values[1][0])
            not_used = np.delete(not_used,np.where(not_used == values[1][0])[0][0])
            indices[values[0][0]] = values[1][0]
        array[values[0][0],values[1][0]] = -1
        values = np.where(array == array.max())

    '''
    for i in range(len(indices)):
        if indices[i] == -1:
            indices[i] = max(indices) + 1
    '''

    return indices,not_used


def match_feature_similarity(older,newer):

    if len(older.id) == 0:
        return newer.id,newer

    if type(newer.id) == type(None):
        newer.id = older.id
        newer.features = older.features
        newer.lives = [x-1 for x in older.lives]
        newer.colours = older.colours
        return newer.id,newer

    old_features = older.features
    old_id = older.id
    new_features = newer.features
    new_id= newer.id

    similarity = np.zeros((max(new_id)+1,max(old_id)+1))


    for old_roi in range(len(old_id)):
        for new_roi in range(len(new_id)):
            similarity[new_id[new_roi],old_id[old_roi]] = np.absolute(np.mean(new_features[new_roi,:,:,:]-old_features[old_roi,:,:,:]))
    #newer.id = np.argmin(similarity[1:,1:],axis = 1)+1

    newer.id = Tracker.max_index_selector(similarity[1:,1:,None])

    for ind in newer.id:
        if ind < len(older.id)-1:
            newer.colours[ind-1] = older.colours[ind-1]
        else:
            continue
    #newer.colours = [older.colours[ind-1] for ind in newer.id if ind<len(older.id)]
    newer.lives = [x+1  if x<7 else x for x in newer.lives]
    return newer.id,newer
