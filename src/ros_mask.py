#!/usr/bin/env python
import os
import sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import random


import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# OpenCV
import cv2
import time
# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage

# Root directory of the project
ROOT_DIR = os.path.join(os.path.abspath("./"),"src","mask_rcnn","src")
VERBOSE=False

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import tensorflow as tf
graph = tf.get_default_graph()
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config



print(os.path.join(ROOT_DIR,"samples","coco"))
sys.path.append(os.path.join(ROOT_DIR,"samples","coco"))  # To find local version
import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class model:
    def __init__(self):

        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera/image_raw",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")

        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        # Directory of images to run detection on
        self.IMAGE_DIR = os.path.join(ROOT_DIR, "images")
        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
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


    def callback(self,ros_data):
        start = time.time()
        '''Callback function of subscribed topic.
        Here images get converted and features detected'''
        if VERBOSE :
            print ('received image of type: "%s"' % ros_data.format)

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:


        # Load a random image from the images folder
        #print(self.IMAGE_DIR)
        file_names = next(os.walk(self.IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(self.IMAGE_DIR, random.choice(file_names)))
        #print(type(image))
        #print(type(image_np))
        # Run detection
        with graph.as_default():
            results = self.model.detect([image_np], verbose=1)

        # Visualize results
        r = results[0]
        print(results[0])
        end = time.time()
        print("Took around : " + str(end-start))
        #visualize.display_instances(image_np, r['rois'], r['masks'], r['class_ids'],self.class_names, r['scores'])



def main(args):
    '''Initializes and cleanup ros node'''
    ic = model()
    rospy.init_node('mask_rcnn', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
