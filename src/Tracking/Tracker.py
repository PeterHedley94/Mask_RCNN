#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
#import camshift



def bb_intersection_over_union(boxA, boxB):
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


def match_ROIs(older, newer, IOU_threshold):

    #NO OBJECTS IN LAST FRAME
    if(older.roi.shape[0] == 0):
        print("No objects in last frame")
        return newer.id, newer

    funcdict = {'bb_intersection_over_union': bb_intersection_over_union}
    matches = np.zeros((older.roi.shape[0],newer.roi.shape[0]))
    indices = np.zeros((newer.roi.shape[0]))

    for old_ in range(0,older.roi.shape[0]):
        for new_ in range(0,indices.shape[0]):
            matches[old_,new_] = funcdict['bb_intersection_over_union'](older.roi[old_], newer.roi[new_])
            #IOU > 0.5 = ROIs MATCH
            if(matches[old_,new_]> IOU_threshold):
                newer.id[new_] = older.id[old_]
                newer.hist[new_] = older.hist[old_]
                if(older.lives[old_] < 7):
                    newer.lives[new_] = older.lives[old_] + 1
                else:
                    newer.lives[new_] = older.lives[old_]
                newer.colours[new_] = older.colours[old_]
                indices[new_] = older.id[old_]

        #CHECK THERE ARE OBJECTS IN NEW FRAME
        if(matches.shape[1] != 0):
            max_val = np.max(matches[old_,:])
        else:
            max_val = 0

        #APPEND OLD OBJECTS TO NEW FRAME
        if (max_val < IOU_threshold and older.lives[old_] > 0):
            newer.roi = np.concatenate((newer.roi, older.roi[None,old_,:]), axis=0)
            newer.id = np.concatenate((newer.id,older.id[old_,None]), axis=0)
            newer.lives.append(older.lives[old_]-1)
            newer.hist.append(older.hist[old_])
            newer.colours.append(older.colours[old_])
            print("Older shape :" + str(older.masks[:,:,old_,None].shape))
            print("Newer shape :" + str(newer.masks.shape))
            if newer.masks.shape[2] == 0:
                newer.masks = older.masks[:,:,old_,None]
                newer.features = older.features[old_,:,:,:]
            else:
                newer.masks = np.concatenate([newer.masks,older.masks[:,:,old_,None]],axis = 2)
                newer.features = np.concatenate([newer.features,older.features[None,old_,:,:,:]],axis = 0)

    #CHECK THERE ARE OBJECTS IN NEW FRAME
    if(matches.shape[1] == 0):
        return [],newer

    #FIND UNMATCHED INDICES AND ASSIGN NEW INDEX
    new_roi_values = np.max(matches,axis=0)
    max_index = max(older.id) + 1
    new_objects = np.where(new_roi_values<0.5)

    for i in list(new_objects)[0]:
        indices[i] = max_index
        newer.id[i] = max_index
        max_index += 1

    return indices,newer


def BRISK_matcher(older, newer):
    print("hello")

def max_index_selector(array):
    already_used = []
    indices = [-1] * array.shape[0]
    max_indices = np.ones(array.shape)
    values = np.where(array == array.min())

    #when found all new indices stop
    while(array.min()!= 1000 and len(already_used)<array.shape[0]):

        if values[1][0] not in already_used and min(max_indices[values[0][0],:]) != 0:
            already_used.append(values[1][0])
            max_indices[values[0][0],values[1][0]] = 0
            indices[values[0][0]] = values[1][0]+1
        array[values[0][0],values[1][0]] = 1000
        values = np.where(array == array.min())

    for i in range(len(indices)):
        if indices[i] == -1:
            indices[i] = max(indices) + 1

    return indices


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
