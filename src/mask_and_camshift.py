#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import time, threading,random, math, numpy as np, skimage.io, matplotlib, matplotlib.pyplot as plt, cv2
# Ros libraries
import roslib, rospy
# Ros Messages
from sensor_msgs.msg import CompressedImage
# Root directory of the project
#RUNNING WITHOUT ROS
ROOT_DIR = os.path.abspath("./mask_rcnn")
#RUNNING WITH ROS
#ROOT_DIR = os.path.join(os.path.abspath("./"),"src","mask_rcnn","src","mask_rcnn")
VERBOSE=False
#image_buffer
from mask_rcnn.image_class import *
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mask_rcnn.mrcnn import utils
import tensorflow as tf
graph = tf.get_default_graph()
import mask_rcnn.mrcnn.model as modellib
from mask_rcnn.mrcnn import visualize

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
        #self.mask_roi_buffer = circular_buffer()
        mask_roi_buffer_lock = threading.Lock()
        self.buffers = {"mask":collections.deque(maxlen=2),"image":collections.deque(maxlen=2),"camshift":collections.deque(maxlen=2)}
        self.buffer_locks = {"mask":threading.Lock(),'camshift':threading.Lock()}
        self.camshift_update = True

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

    def camshift(self,frame):
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        while(1):
            #UPDATE CAMSHIFT BUFFER WITH NEW VALUES FROM SEG THREAD
            self.buffer_locks["camshift"].acquire()
            if(self.camshift_update):
                track(self,"camshift")
                self.camshift_update = False

            image = self.buffers["image"][0]

            for roi_index in self.buffer_locks["camshift"][0].roi:

                old_roi = self.buffer_locks["camshift"][0].roi[roi_index]
                ''''ry1, rx1, ry2, rx2'''
                r,h = old_roi[2],old_roi[4]-old_roi[2]
                c,w = old_roi[1],old_roi[3]-old_roi[1]
                track_window = (c,r,w,h)

                roi = frame[r:r+h, c:c+w]
                hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

                hsv = cv2.cvtColor(image.frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                # apply meanshift to get the new location
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                self.buffer_locks["camshift"][0].roi[roi_index] = [c,c+w,r,r+h]

            self.buffer_locks["camshift"].release()
            return pts

    def bb_intersection_over_union(self,boxA, boxB):
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

    def match_ROIs(self,older, newer):

        #NO OBJECTS IN LAST FRAME
        if(older.roi.shape[0] == 0):
            print("No objects in last frame")
            return newer.id, newer

        matches = np.zeros((older.roi.shape[0],newer.roi.shape[0]))
        indices = np.zeros((newer.roi.shape[0]))

        for old_ in range(0,older.roi.shape[0]):
            for new_ in range(0,indices.shape[0]):
                matches[old_,new_] = self.bb_intersection_over_union(older.roi[old_], newer.roi[new_])
                #IOU > 0.5 = ROIs MATCH
                if(matches[old_,new_]>0.5):
                    newer.id[new_] = older.id[old_]
                    indices[new_] = older.id[old_]

            #CHECK THERE ARE OBJECTS IN NEW FRAME
            if(matches.shape[1] != 0):
                max_val = np.max(matches[old_,:])
            else:
                max_val = 0

            #APPEND OLD OBJECTS TO NEW FRAME
            if (max_val < 0.5 and older.lives[old_] > 0):
                newer.roi = np.concatenate((newer.roi, older.roi[None,old_,:]), axis=0)
                newer.id = np.concatenate((newer.id,older.id[old_,None]), axis=0)
                newer.lives.append(older.lives[old_]-1)

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

    def track(self,type_):

        buff_size = len(self.buffers[type_])

        if buff_size == 1 or buff_size == 0:
            return self.buffers[type_][0].id

        older = self.buffers[type_].popleft()
        newer = self.buffers[type_][0]
        [indices,newer] = self.match_ROIs(older,newer)
        self.buffers[type_][0] = newer

        return indices


    def mask_predict(self):

        while(1):
            try:
                image = self.buffers["image"][0]
            except IndexError:
                image = False
            if(image != False):
                start = time.time()
                with graph.as_default():
                    #results contains ['rois', 'scores', 'class_ids', 'masks']
                    results = self.model.detect([image.frame], verbose=1)

                # Visualize results
                r = results[0]
                res_ = roi_class(r['rois'],image.time)

                self.buffer_locks["mask"].acquire()

                self.buffers['mask'].append(res_)
                #TRACK from LAST Mask position
                r['ids'] = self.track('mask')

                self.buffer_locks["camshift"].acquire()
                self.buffers["camshift"].append(self.buffers['mask'][1])
                self.camshift_update = True
                self.buffer_locks["camshift"].release()

                self.buffer_locks["mask"].release()
                #print(results)
                end = time.time()
                print("Took around : " + str(end-start))
                    #r['class_ids']
                visualize.display_instances(image.frame, r['rois'], r['masks'],r['class_ids'], self.class_names, r['ids'], r['scores'])
            else:
                time.sleep(1)



    def callback(self,frame):#ros_data):
        frame_time = time.time()
        image = image_class(frame,frame_time)
        self.buffers["image"].append(image)
        #self.mask_predict()


def main(args):
    '''Initializes and cleanup ros node'''
    ic = model()

    mask_rcnn_thread = threading.Thread(target=ic.mask_predict)
    mask_rcnn_thread.daemon = True
    mask_rcnn_thread.start()

    #rospy.init_node('mask_rcnn', anonymous=True)
    try:
        #rospy.spin()
        print(os.getcwd())
        cap = cv2.VideoCapture('vid.mp4')
        if(cap.open(0) == False):
          print("Failed to get camera")
        else:
            print("Located camera")
        # take first frame of the video
        while(1):
            ret,frame = cap.read()
            if ret ==True:
                ic.callback(frame)
            else:
                print("Failed")
                continue



    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
