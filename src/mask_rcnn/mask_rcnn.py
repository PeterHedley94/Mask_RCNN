#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
ROOT_DIR = os.path.abspath(".")
print(ROOT_DIR)
VERBOSE=False
sys.path.append(ROOT_DIR)
from mrcnn import utils
import tensorflow as tf
graph = tf.get_default_graph()
import mrcnn.model as modellib
from mrcnn import visualize
import pickle,time, threading,random, math, numpy as np, skimage.io, matplotlib, matplotlib.pyplot as plt, cv2


# Import COCO config

print(os.path.join(ROOT_DIR,"mask_rcnn","samples","coco"))
sys.path.append(os.path.join(ROOT_DIR,"mask_rcnn","samples","coco"))  # To find local version
import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class model:
    def __init__(self):

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

        self.initialise_mrcnn()


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

    def remove_indices(self,r,indices):
        r['rois'] = r['rois'][np.where(indices)]
        r['scores'] = r['scores'][np.where(indices)]
        r['class_ids'] = r['class_ids'][np.where(indices)]
        r['masks'] = r['masks'][:,:,np.where(indices)[0]]
        return r


    def remove_classes(self,r):
        indices = [True if x in self.class_indices else False for x in r['class_ids']]
        return self.remove_indices(r,indices)


    def remove_zero_area(self,r):
        indices = [False if roi[2]-roi[0]==0 or roi[3]-roi[1]==0 else True for roi in r['rois']]
        return self.remove_indices(r,indices)



    def mask_predict(self,image):
        filename = str(image.time) + ".p"
        directory = "tate3_masks"
        if not os.path.exists(os.path.join(ROOT_DIR,directory,filename)):
            with graph.as_default():
                #results contains ['rois', 'scores', 'class_ids', 'masks']
                results = self.model_.detect([image.frame], verbose=0)
            # Put resuts in buffer
            r = results[0]
            self.remove_classes(r)
            self.remove_zero_area(r)
            pickle.dump(r, open(os.path.join(ROOT_DIR,directory,filename), "wb" ))
        else:
            r = pickle.load( open(os.path.join(ROOT_DIR,directory,filename), "rb" ) )
            
        return [r,visualize.random_colors(r['rois'].shape[0])]
