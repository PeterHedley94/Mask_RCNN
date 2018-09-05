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

def tracker_predict(older,old_,newer,new_):
    bbox = older.tracker_predictions[old_]
    if bbox == [0,0,0,0]:
        return 0.1
    probability = multivariate_normal.pdf(newer.roi_dims_c[[0,1,4,5],new_],mean=bbox, cov=40*np.eye(4))
    return probability

def tracker_predict_probability(array):
    return array

#Scale Kalman output
def Kalman_predict_probability(array):
    global KALMAN_LIKELIHOOD_THRESHOLD
    array = KALMAN_LIKELIHOOD_THRESHOLD - array
    array[array < 0] = 0
    return array

#Return % of mean values of new box is at compared to predicted values
#could be improved
def Kalman_predict(older,old_,newer,new_):
    state = np.array(newer.roi_dims_w[[0,1,2,4,5],new_])
    return older.kalman[old_].get_log_likelihood(state[:,None])



def match_ROIs(older, newer, BRISK_):

    #NO OBJECTS IN LAST FRAME
    if(older.no_rois == 0):
        print("No objects in last frame")
        return newer.id, newer

    funcdict = [tracker_predict,BRISK_.get_match_score,Kalman_predict]
    funcprob = [tracker_predict_probability,BRISK_.get_probabilities,Kalman_predict_probability]

    matches = np.zeros((len(funcdict),newer.no_rois,older.no_rois))
    indices = np.zeros((newer.no_rois))

    for old_ in range(0,older.no_rois):
        older.kalman[old_].predict()
        older.intialise_tracker(old_)

        try:
            _,older.tracker_predictions[old_] = older.tracker[old_].update(newer.image)
        except:
            older.tracker_predictions[old_] = older.roi_dims_c[[0,1,4,5],old_].tolist()
        for new_ in range(0,indices.shape[0]):
            class_match = match_classes(older,old_,newer,new_)
            for c,method_ in enumerate(funcdict):
                if class_match:
                    matches[c,new_,old_] = method_(older,old_,newer,new_)
                else:
                    matches[c,new_,old_] = 0
    start = time.time()

    for c,method_ in enumerate(funcprob):
        matches[c,:,:] = method_(matches[c,:,:])
        if c > 0:
            matches[0,:,:] = np.multiply(matches[0,:,:],matches[c,:,:])

    if newer.no_rois > 0:
        [final_indices,not_in] = max_index_selector(matches[0,:,:])
    else:
        final_indices = []
        not_in = range(len(older.id))

    newer.append_rois(older,not_in)
    newer.match_old_rois(older,final_indices)
    newer.id_new_objects(final_indices)
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
