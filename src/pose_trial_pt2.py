

#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np, cv2
#RUNNING WITHOUT ROS
ROOT_DIR = os.path.abspath("./mask_rcnn")
import json
import matplotlib.pyplot as plt
from utils.folder_manipulation import *
from mpl_toolkits import mplot3d


def to_number(item):
    item = item.split("/")[-1]
    number = int(item.split("_")[0])
    return number


def gd(filename):
    return np.loadtxt(filename)

def main(args):
    '''Initializes and cleanup ros node'''
    path = "/home/peter/Documents/okvis_drl/build/buck_imp1_dataset"#"/home/peter/Tests"
    #path = '/home/peter/catkin_ws/src/mask_rcnn/src/mask_rcnn/at.avi'

    #images = get_image_names(path+"/cam0/data")
    #depth_images = get_xml_names(path+"/cam1/data")
    o_T_WS_C = get_file_names(os.path.join(path,"pose"),"T_WS_C.txt",to_number)
    o_T_WS_r = get_file_names(os.path.join(path,"pose"),"T_WS_r.txt",to_number)
    o_sb = get_file_names(os.path.join(path,"pose"),"sb.txt",to_number)

    #cap = cv2.VideoCapture(path)
    #frame_width = get_image(images[0]).shape[1]
    #frame_height = get_image(images[0]).shape[0]

    with open(os.path.join("utils",'camera_model.json')) as f:
        camera_model = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    #ax = fig.add_subplot(111)
    plt.gca().set_aspect('equal', adjustable='box')
    xpoints = []
    ypoints = []
    zpoints = []


    for T_WS_r in o_T_WS_r:

        data = gd(T_WS_r)
        xpoints.append(data[0])
        ypoints.append(data[1])
        zpoints.append(data[2])
        #print("###############################")
    ax.plot3D(xpoints,ypoints,zpoints)
    print(len(xpoints))
    #ax.plot(xpoints,ypoints)
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
