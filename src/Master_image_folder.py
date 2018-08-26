#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import time,json,threading,random, math, numpy as np, skimage.io, matplotlib, matplotlib.pyplot as plt, cv2
#RUNNING WITHOUT ROS
ROOT_DIR = os.path.abspath("./mask_rcnn")
#IMAGE_COUNT = 0
#RUNNING WITH ROS
#ROOT_DIR = os.path.join(os.path.abspath("./"),"src","mask_rcnn","src","mask_rcnn")
VERBOSE=False
#image_buffer
from utils.image_class import *
import pickle
from utils.folder_manipulation import *
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

#NEW IMPORTS
from mask_rcnn import *
from utils.utils import print_progress
from utils.visualiser import *
from Tracking.Tracker import *
from Tracking.BRISK import *
from MovementModels.cyclist_model import *


IMAGE_TIME = 0

class cycle_safe:
    def __init__(self,im_width,im_height,camera_model):

        #Create buffers for threads
        mask_roi_buffer_lock = threading.Lock()
        self.buffers = {"mask":collections.deque(maxlen=2),"mask_image":collections.deque(maxlen=2),"image":collections.deque(maxlen=2),"camshift":collections.deque(maxlen=2)}
        self.buffer_locks = {"mask":threading.Lock(),'camshift':threading.Lock()}
        self.count = 0
        self.camera_model = camera_model
        # Define the codec and create VideoWriter object
        self.output_width = im_width
        self.output_height = im_height
        self.output_writers = {"combined":cv2.VideoWriter_fourcc(*'XVID')}
        self.output_videos = {"combined":cv2.VideoWriter('combined.avi',self.output_writers["combined"], 30.0, (self.output_width,self.output_height))}
        self.camshift_update = False
        self.stop = False
        self.model = model()
        self.BRISK_ = BRISK_class()
        self.visualiser_ = visualiser(self.output_width,self.output_height)
        self.cm = cycle_model()


    def track(self,type_):

        buff_size = len(self.buffers[type_])
        if buff_size == 1:
            return self.buffers[type_][0].id
        elif buff_size == 0:
            return False

        older = self.buffers[type_][-2]
        newer = self.buffers[type_][-1]
        [indices,newer] = match_ROIs(older,newer, self.BRISK_)
        self.buffers[type_][0] = newer
        return indices


    def undistort_image(self,image):
        mtx = np.zeros((3,3))
        mtx[0,0],mtx[1,1] = 1/self.camera_model["focal_length"][0],1/self.camera_model["focal_length"][1]
        mtx[0,1],mtx[0,2] = self.camera_model["principal_point"]
        mtx[2,2] = 1
        dist = np.array(self.camera_model["distortion_coefficients"])
        w,h = image.shape[:2]
        #newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        new_image = cv2.undistort(image, mtx, dist, None,None)# newcameramtx)
        return new_image


    def callback(self,i,d,T_WS_C,T_WS_r,sb,camera_model,time):#ros_data):
        undist_image = i#self.undistort_image(i)
        image = image_class(undist_image,d,time)#,frame_time)
        self.buffers["image"].append(image)
        r,colours = self.model.mask_predict(image)

        des,key = self.BRISK_.get_des_key(image.frame,r['rois'],r['masks'])
        res_ = roi_class(r['rois'],image.time,r['class_ids'],
        colours,r['masks'],des=des,key=key,image=image.frame,depth = d,pose =[T_WS_C,T_WS_r,sb],camera_model=self.camera_model)

        self.buffers['mask'].append(res_)
        self.track('mask')

        self.cm.add_points([T_WS_C,T_WS_r,sb],time)
        self.visualiser_.write_to_video(self.output_videos['combined'],self.buffers['mask'][-1],
        self.model.class_names,T_WS_r,T_WS_C,camera_model,self.cm)


def to_number(item):
    item = item.split("/")[-1]
    number = int(item.split("_")[0])
    return number

def gd(filename):
    return np.loadtxt(filename)

def get_time_from_filename(filename):
    return float(filename.split("/")[-1].split(".")[0])

def get_nearest_pose(time,filenames):
    time = get_time_from_filename(time)
    for i,name in enumerate(filenames):
        val = float(name.split("/")[-1].split("_")[0])
        if val >= time and i==0:
            return i
        elif val>= time:
            if val-time > time- float(filenames[-1].split("/")[-1].split("_")[0]):
                return i-1
            else:
                return i
    return len(filenames)-1

def main(args):
    '''Initializes and cleanup ros node'''
    path = '/home/peter/Documents/okvis_drl/build/tate3_dataset'#tate3_dataset'#blackfriars1_dataset'#
    #path = '/home/peter/catkin_ws/src/mask_rcnn/src/mask_rcnn/at.avi'

    images = get_image_names(path+"/cam0/data")
    print(images)
    depth_images = get_xml_names(path+"/cam1/data")
    o_T_WS_C = get_file_names(os.path.join(path,"pose"),"T_WS_C.txt",to_number)
    o_T_WS_r = get_file_names(os.path.join(path,"pose"),"T_WS_r.txt",to_number)
    o_sb = get_file_names(os.path.join(path,"pose"),"sb.txt",to_number)

    #cap = cv2.VideoCapture(path)
    frame_width = 480#get_image(images[0]).shape[1]
    frame_height = 180*3#get_image(images[0]).shape[0]*2

    print("Frame width is "  + str(frame_width))
    print("Frame height is "  + str(frame_height))

    with open(os.path.join("utils",'camera_model.json')) as f:
        camera_model = json.load(f)

    ic = cycle_safe(frame_width,frame_height,camera_model)

    for index,i,d in zip(range(len(images)),images,depth_images):
        print("Index is " + str(index))
        if index > 0:#index > 87 and index < 89: #> 1570 and index <1576:
            pose = get_nearest_pose(i,o_T_WS_r)
            T_WS_C,T_WS_r,sb = o_T_WS_C[pose],o_T_WS_r[pose],o_sb[pose]
            ic.callback(get_image(i),get_array_xml(d),gd(T_WS_C),gd(T_WS_r),gd(sb),camera_model,get_time_from_filename(i))
            print("###############################")


if __name__ == '__main__':
    main(sys.argv)
