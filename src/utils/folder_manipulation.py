#!/usr/bin/env python
'''
FILE FOR BASIC FOLDER AND FILE FOLDER MANIPULATION
'''
import os
import numpy as np
import xml.etree.ElementTree as ET
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import cv2


#RETURNS ALL FOLDER NAMES IN A DIRECTORY
def get_folders(directory_):
    items = os.listdir(directory_)
    folder_present = False
    folders = []
    for d in items:
        if os.path.isdir(os.path.join(directory_, d)):
            folders.append(d)
            folder_present = True

    if not folder_present:
        print("Could not find any folders/categories!")
    return sorted(folders)

def delete_files(files):
    for file in files:
        ## If file exists, delete it ##
        if os.path.isfile(file):
            os.remove(file)
        else:    ## Show an error ##
            print("Error: %s file not found" % file)

#RETURNS ALL IMAGES NAMES IN A DIRECTORY
def get_file_names(directory_,string,key = None):
    files = os.listdir(directory_)
    image_present = False
    images = []
    for file in files:
        if file.endswith(string):
            images.append(os.path.join(directory_,file))
            image_present = True

    if not image_present:
        print("Could not find any Files!")
    images.sort(key=key)
    return images


#RETURNS ALL YAML NAMES IN A DIRECTORY
def get_xml_names(directory_):
    files = os.listdir(directory_)
    image_present = False
    images = []
    for file in files:
        if file.endswith('.xml'):
            images.append(os.path.join(directory_,file))
            image_present = True

    if not image_present:
        print("Could not find any xml files!")
    images.sort()
    return images


def get_array_xml(file):
    tree = ET.parse(file)
    rows = tree.find("depth_image").find("rows").text
    cols = tree.find("depth_image").find("cols").text
    array = np.fromstring(tree.find("depth_image").find("data").text,sep = " ",dtype=np.uint16)
    return np.reshape(array,(int(rows),int(cols)))

#RETURNS ALL IMAGES NAMES IN A DIRECTORY
def get_image_names(directory_):
    files = os.listdir(directory_)
    image_present = False
    images = []
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            images.append(os.path.join(directory_,file))
            image_present = True

    if not image_present:
        print("Could not find any Images!")
    images.sort()
    return images

#GET AN IMAGE FROM FILE
def get_image(filepath):
    img = cv2.imread(filepath)
    #resized_image = np.expand_dims(cv2.resize(img, (IM_HEIGHT, IM_WIDTH)), axis=0)
    return img


#GET AN IMAGE FROM FILE
def get_resized_image(filepath,height,width):
    img = cv2.imread(filepath)
    resized_image = cv2.resize(img, (width,height)) #np.expand_dims(, axis=0)
    return resized_image

#STACK ALL IMAGES IN A FOLDER INTO AN TENSOR
def dstack_folder_sequence(directory_, sequence_length):
    image_list = get_image_names(directory_)
    if len(image_list) != sequence_length:
        for i in range(len(image_list),sequence_length,1):
            image_list.append(image_list[len(image_list)-1])

    if sequence_length != 1 and len(image_list) != sequence_length:
        print("Incompatible sequence {}".format(directory_))
        return

    images = get_image(os.path.join(directory_, image_list[0]))
    if len(image_list) > 1:
        for image in image_list[1:]:
            new_image = get_image(os.path.join(directory_, image))
            images = np.concatenate([new_image,images],axis = 0)
    return images


# def dstack_folder(directory_):
#     image_list = get_image_names(directory_)
#     images = get_image(os.path.join(directory_, image_list[0]))
#     if len(image_list) > 1:
#         for image in image_list[1:]:
#             new_image = get_image(os.path.join(directory_, image))
#             images = np.concatenate([new_image,images],axis = 0)
#     return images
