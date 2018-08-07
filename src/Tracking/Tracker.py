#!/usr/bin/env python
import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import math, numpy as np,time
np.set_printoptions(precision=3)
from scipy.stats import multivariate_normal
from Tracking.BRISK import *
from Tracking.Kalman import *
from utils.common import *
from collections import OrderedDict


#Check if the classes match
def match_classes(older,old_,newer,new_):
    if older.class_[old_] == newer.class_[new_]:
        return True
    else:
        return False

#Scale Kalman output
def Kalman_predict_probability2(array):
    return array *0.9

#Return % of mean values of new box is at compared to predicted values
#could be improved
def Kalman_predict2(older,old_,newer,new_):
    p = older.kalman[old_].statePre
    cov_ = older.kalman[old_].errorCovPost
    #get residual consistency
    #s residual covariance matrix, y is the residual
    probability = multivariate_normal.pdf([newer.centre_x[new_],newer.centre_y[new_],newer.roi_width[new_],newer.roi_height[new_]],
        mean=p[:4,0], cov=cov_[:4,:4])
    probability_at_mean = multivariate_normal.pdf(p[:4,0],mean=p[:4,0], cov=cov_[:4,:4])
    return probability/probability_at_mean

def tracker_predict(older,old_,newer,new_):
    bbox = older.tracker_predictions[old_]
    if bbox == [0,0,0,0]:
        return 0.1
    probability = multivariate_normal.pdf([newer.centre_x[new_],newer.centre_y[new_],newer.roi_width[new_],newer.roi_height[new_]],
        mean=bbox, cov=40*np.eye(4))
    return probability

def tracker_predict_probability(array):
    return array

#Scale Kalman output
def Kalman_predict_probability(array):
    global KALMAN_LIKELIHOOD_THRESHOLD
    #print("Array to kalman prob is ")
    #print(str(array))
    array = KALMAN_LIKELIHOOD_THRESHOLD - array
    array[array < 0] = 0
    #print("Array out of kalman prob is ")
    #print(str(array))
    return array

#Return % of mean values of new box is at compared to predicted values
#could be improved
def Kalman_predict(older,old_,newer,new_):
    state = np.array([newer.centre_x[new_],newer.centre_y[new_],newer.roi_width[new_],newer.roi_height[new_]])
    #print("New state is : " + str(state) + " vs predicted state : " + str(older.kalman[old_].statePre[:4,:]))
    return older.kalman[old_].get_log_likelihood(state[:,None])

#Add older ROIs that were not found in current frame
def append_old_rois_not_in_frame(older,newer,not_in):
    for old_ in not_in:
        if older.lives[old_]-1 > 0:

            newer.roi = np.concatenate((newer.roi, older.roi[None,old_,:]), axis=0)
            newer.id = np.concatenate((newer.id,older.id[old_,None]), axis=0)
            newer.class_ = np.concatenate((newer.class_,older.class_[old_,None]),axis = 0)
            newer.kalman.append(older.kalman[old_])

            predictions = older.kalman[old_].statePre
            newer.centre_x.append(predictions[0])
            newer.centre_y.append(predictions[1])
            newer.roi_width.append(predictions[2])
            newer.roi_height.append(predictions[3])

            newer.tracker.append(older.tracker[old_])
            newer.tracker_predictions.append([0,0,0,0])

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
    return newer

def match_ROIs(older, newer, BRISK_):

    #NO OBJECTS IN LAST FRAME
    if(older.roi.shape[0] == 0):
        print("No objects in last frame")
        return newer.id, newer

    #Methods of matching ROIs
    #funcdict = [BRISK_.get_match_score,Kalman_predict]
    #funcdict = [Kalman_predict]
    print("Older roi _length " + str(older.roi.shape[0]))
    print("Older tracker_length " + str(len(older.tracker)))
    funcdict = [tracker_predict]
    #'bb_intersection_over_union': bb_intersection_over_union,
    #funcprob = [BRISK_.get_probabilities,Kalman_predict_probability]
    #funcprob = [Kalman_predict_probability]
    funcprob = [tracker_predict_probability]
    #'bb_intersection_over_union': bb_intersection_over_union_probabilities,

    matches = np.zeros((len(funcdict),newer.roi.shape[0],older.roi.shape[0]))
    indices = np.zeros((newer.roi.shape[0]))
    #print("Order is " + str(funcdict.values()))
    #print("Order is " + str(funcprob.values()))
    #Score the match for each old ROI to each new ROI

    for old_ in range(0,older.roi.shape[0]):
        older.kalman[old_].predict()

        #if older.tracker[old_] == None:
        older.intialise_tracker(old_)

        _,older.tracker_predictions[old_] = older.tracker[old_].update(newer.image)
        for new_ in range(0,indices.shape[0]):
            class_match = match_classes(older,old_,newer,new_)
            for c,method_ in enumerate(funcdict):
                if class_match:
                    matches[c,new_,old_] = method_(older,old_,newer,new_)
                else:
                    matches[c,new_,old_] = 0
    start = time.time()

    #print(matches)
    #print(".............")
    #Combine methods
    for c,method_ in enumerate(funcprob):
        matches[c,:,:] = method_(matches[c,:,:])
        if c > 0:
            #matches[0,:,:] = matches[0,:,:] + matches[c,:,:]
            matches[0,:,:] = np.multiply(matches[0,:,:],matches[c,:,:])
    #print(matches)
    #print(".............")
    #print("MAx index theshold is " + str(INDEX_SELECTOR_THRESHOLD))
    #Check there are detected objects in new frame
    #print(matches)
    #print(".............")
    if newer.roi.shape[0] > 0:
        [final_indices,not_in] = max_index_selector(matches[0,:,:])
    else:
        final_indices = []
        not_in = range(len(older.id))

    newer = append_old_rois_not_in_frame(older,newer,not_in)

    #Add objects detected in both frames
    for c,index in enumerate(final_indices):
        if index < len(older.id) and index >= 0:
            newer.id[c] = older.id[index]
            newer.colours[c] = older.colours[index]
            new_state = np.array([newer.centre_x[c],newer.centre_y[c],newer.roi_width[c],newer.roi_height[c]],dtype = np.float64)
            newer.kalman[c] = older.kalman[index]
            newer.kalman[c].correct(new_state)
            #newer.tracker[c] = older.tracker[index]

            if older.lives[index] < 7:
                newer.lives[c] = older.lives[index] +1

    if len(newer.id) > 0:
        max_id = max(newer.id)+1
    else:
        max_id = 0
    #Add objects that only appear in new frame
    for c,index in enumerate(final_indices):
        if index == -1:
            newer.id[c] = max_id
            max_id += 1

    if len(newer.id) != len(set(newer.id)):
        print("THERE ARE DUPLICATE IDS")
    return newer.id,newer


def max_index_selector(array):
    global INDEX_SELECTOR_THRESHOLD
    already_used = []
    not_used = np.arange(array.shape[1])
    indices = [-1] * array.shape[0]
    values = np.where(array == array.max())

    #when found all new indices stop
    while(array.max() > INDEX_SELECTOR_THRESHOLD and len(already_used) < array.shape[0]):
        if values[1][0] not in already_used and indices[values[0][0]] == -1:
            already_used.append(values[1][0])
            not_used = np.delete(not_used,np.where(not_used == values[1][0])[0][0])
            indices[values[0][0]] = values[1][0]
        array[values[0][0],values[1][0]] = -1
        values = np.where(array == array.max())

    return indices,not_used

'''

#Scale iou confidence
def bb_intersection_over_union_probabilities(array):
    return array * 0.9

#Return intersection over union area
def bb_intersection_over_union(older,old_,newer,new_):
    boxA = older.roi[old_]
    boxB = newer.roi[new_]

    ff=1
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

    if(float(boxAArea + boxBArea - interArea) > 0):
        iou = abs(interArea / float(boxAArea + boxBArea - interArea))
    else:
        iou = 0
    # return the intersection over union value
    return iou



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
'''
