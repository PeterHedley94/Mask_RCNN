#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import time, threading,random, math, numpy as np, skimage.io, matplotlib, matplotlib.pyplot as plt, cv2
# Ros libraries
import roslib, rospy,imutils
# Ros Messages
from sensor_msgs.msg import Image, CompressedImage
# Root directory of the project
#RUNNING WITHOUT ROS
#print(ROOT_DIR)
#ROOT_DIR = os.path.abspath("./mask_rcnn")
#IMAGE_COUNT = 0
'''RUNNING WITH ROS'''
ROOT_DIR = os.path.join(os.path.abspath("./"),"src","mask_rcnn","src")
print(ROOT_DIR)
VERBOSE=False
#image_buffer
from utils.image_class import *
import pickle
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mask_rcnn.mrcnn import utils
import tensorflow as tf
graph = tf.get_default_graph()
import mask_rcnn.mrcnn.model as modellib
from mask_rcnn.mrcnn import visualize

# Import COCO config

print(os.path.join(ROOT_DIR,"mask_rcnn","samples","coco"))
sys.path.append(os.path.join(ROOT_DIR,"mask_rcnn","samples","coco"))  # To find local version
import coco

#NEW IMPORTS

from utils.utils import print_progress
from utils.visualiser import *
from Tracking.Tracker import *
from Tracking.BRISK import *

IMAGE_TIME = 0


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class model:
    def __init__(self,im_width,im_height):

        #Create buffers for threads
        mask_roi_buffer_lock = threading.Lock()
        self.buffers = {"mask":collections.deque(maxlen=2),"mask_image":collections.deque(maxlen=2),"image":collections.deque(maxlen=1),"camshift":collections.deque(maxlen=2)}
        self.buffer_locks = {"mask":threading.Lock(),'camshift':threading.Lock()}
        self.count = 0

        # Define the codec and create VideoWriter object
        self.output_width = im_width
        self.output_height = im_height
        self.output_writers = {"combined":cv2.VideoWriter_fourcc(*'XVID')}
        self.output_videos = {"combined":cv2.VideoWriter('combined.avi',self.output_writers["combined"], 20.0, (self.output_width,self.output_height))}
        self.camshift_update = False
        self.stop = False
        self.IOU_threshold = 0.3

        self.model_,self.class_indices,self.class_names = None,None,None

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_indices = [1,2,3,4,6,8]
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

        self.BRISK_ = BRISK_class()

        self.initialise_ros()
        self.initialise_mrcnn()

    def track(self,type_):

        buff_size = len(self.buffers[type_])

        if buff_size == 1:
            return self.buffers[type_][0].id
        elif buff_size == 0:
            return False

        older = self.buffers[type_][-2]
        newer = self.buffers[type_][-1]
        [indices,newer] = match_ROIs(older,newer, self.BRISK_ ,IOU_threshold=0.25)
        #[indices,newer] = self.match_feature_similarity(older,newer)
        #newer.id = np.arange(len(newer.id)) +1
        self.buffers[type_][-2] = newer

        return indices


    def initialise_mrcnn(self):
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self.model_ = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model_.load_weights(COCO_MODEL_PATH, by_name=True)




    def initialise_ros(self):
        '''Initialize ros publisher, ros subscriber'''
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber_image = rospy.Subscriber("/zr300_node/color/image_raw",
            Image, self.callback,  queue_size = 1)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")

    def remove_classes(self,r):
        indices = [True if x in self.class_indices else False for x in r['class_ids']]
        r['rois'] = r['rois'][np.where(indices)]
        r['scores'] = r['scores'][np.where(indices)]
        r['class_ids'] = r['class_ids'][np.where(indices)]
        r['masks'] = r['masks'][:,:,np.where(indices)[0]]

    def mask_predict(self):
        global ROOT_DIR
        try:
            image = self.buffers["image"][0]
        except IndexError:
            image = False
        if(image != False):
            start = time.time()

            '''
            with graph.as_default():
                #results contains ['rois', 'scores', 'class_ids', 'masks']
                results = self.model_.detect([image.frame], verbose=0)
            # Put resuts in buffer
            r = results[0]
            self.remove_classes(r)
            '''
            filename = str(image.time.secs) + "_" +  str(image.time.nsecs) + ".p"
            print("Current Directory is : " + os.getcwd())
            #pickle.dump(r, open(os.path.join(ROOT_DIR,"masks",filename), "wb" ))
            r = pickle.load( open(os.path.join(ROOT_DIR,"masks",filename), "rb" ) )
            des,key = self.BRISK_.get_des_key(image.frame,r['rois'],r['masks'])
            res_ = roi_class(r['rois'],image.time,r['class_ids'],
            visualize.random_colors(r['rois'].shape[0]),r['masks'],r['features'],des=des,key=key)
            #print("Masks Shape is: " + str(len(res_.masks)))
            self.buffer_locks["mask"].acquire()
            self.buffers['mask'].append(res_)
            #TRACK from LAST Mask position
            r['ids'] = self.track('mask')
            self.buffer_locks["mask"].release()
            #print(results)'''
            end = time.time()
            print("Took around : " + str(end-start))


    def callback(self,ros_data):
        global IMAGE_TIME
        #print(ros_data.header.stamp)
        #frame_time = ros_data.header.stamp.secs + ros_data.header.stamp.nsecs/10**9
        np_arr = np.fromstring(ros_data.data, np.uint8)
        np_arr = np.reshape(np_arr,(ros_data.height,ros_data.width,3),order = 'C')
        img = np_arr[...,::-1]

        image = image_class(img,ros_data.header.stamp)
        IMAGE_TIME += 1
        self.buffers["image"].append(image)
        self.mask_predict()
        write_to_video(self.output_videos['combined'],self.buffers['image'][-1].frame,self.buffers['mask'][-2],self.class_names,[self.output_width,self.output_height])

def main(args):
    rospy.init_node('mask_rcnn', anonymous=True)
    frame_width,frame_height = 640,480
    ic = model(frame_width,frame_height)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
        ic.stop = True

    print ("Shutting down ROS Image feature detector module")
    ic.output_videos['combined'].release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
