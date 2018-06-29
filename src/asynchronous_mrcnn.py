#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import time, threading,random, math, numpy as np, skimage.io, matplotlib, matplotlib.pyplot as plt, cv2
# Ros libraries
import roslib, rospy,imutils
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

#CV2 stuff
#cv2.namedWindow('camshift', cv2.WINDOW_NORMAL)
#cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
#cv2.namedWindow('raw', cv2.WINDOW_NORMAL)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class model:
    def __init__(self):
        #self.mask_roi_buffer = circular_buffer()
        mask_roi_buffer_lock = threading.Lock()
        self.buffers = {"mask":collections.deque(maxlen=2),"mask_image":collections.deque(maxlen=2),"image":collections.deque(maxlen=2),"camshift":collections.deque(maxlen=2)}
        self.buffer_locks = {"mask":threading.Lock(),'camshift':threading.Lock()}
        self.count = 0
        # Define the codec and create VideoWriter object
        self.output_width = 640
        self.output_height = 480
        self.output_writers = {"combined":cv2.VideoWriter_fourcc(*'XVID')}
        self.output_videos = {"combined":cv2.VideoWriter('combined.avi',self.output_writers["combined"], 20.0, (640,480))}
        self.camshift_update = False
        self.stop = False

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

    def camshift(self):
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        print("performing camshift")
        count = 1
        while(1):#self.stop == False):
            count += 1
            start = time.time()
            if(len(self.buffers["camshift"]) > 0):
                #UPDATE CAMSHIFT BUFFER WITH NEW VALUES FROM SEG THREAD
                self.buffer_locks["camshift"].acquire()
                if(self.camshift_update):
                    self.track("camshift")
                    self.camshift_update = False

                image = self.buffers["image"][0]

                for roi_index in range(len(self.buffers["camshift"][0].roi)):

                    new_roi = self.buffers["camshift"][0]
                    old_roi = new_roi.roi[roi_index]
                    hist = self.buffers["camshift"][0].hist[roi_index]

                    #print(hist)
                    ''''ry1, rx1, ry2, rx2'''
                    x1,x2 = min(old_roi[1],old_roi[3]),max(old_roi[1],old_roi[3])
                    y1,y2 = min(old_roi[2],old_roi[0]),max(old_roi[2],old_roi[0])

                    #get coords of top edge,height, left edge, width
                    r,h = y1,y2-y1
                    c,w = x1,x2-x1

                    #reduce box size to focus on peron/object trunk
                    div = 3
                    r,h,c,w = int(r+h/2-h/div),int(h/div),int(c+w/2-w/div),int(w/div)

                    track_window = (c,r,w+1,h+1)

                    print("points are : [()" + str(old_roi[1]) + ',' + str(old_roi[0]) + '),(' + str(old_roi[3]) + ',' + str(old_roi[2]) + ")]")
                    print("initial track_window is : " + str(track_window))
                    print("image size is " + str(image.frame.shape))

                    roi = image.frame[r:r+h, c:c+w]
                    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    if all([ v > 0 for v in roi.shape ]):
                        print("roi shape is : " + str(roi.shape))
                        mask = cv2.inRange(hsv_roi, np.array([0., 60.,32.]), np.array([180.,255.,255.]))

                        if(type(hist) != np.ndarray):
                            hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                            new_roi.hist[roi_index] = hist

                        cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)

                        hsv = cv2.cvtColor(image.frame, cv2.COLOR_BGR2HSV)
                        dst = cv2.calcBackProject([hsv],[0],hist,[0,180],1)
                        # apply meanshift to get the new location
                        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                        print("New track window is "  + str(track_window))
                        [c,r,w,h] = track_window

                        #re-calculate box
                        r,h,c,w = int(r+(h)-(h*div/2)),h*div,int(c+w-(w*div/2)),w*div
                        new_roi.roi[roi_index] = [c,c+w,r,r+h]
                        self.buffers["camshift"].append(new_roi)
                        #self.track("camshift")

                self.buffer_locks["camshift"].release()
                end = time.time()
                print("Camshift took around : " + str(end-start))
                time.sleep(0.1)
            else:
                print("Buffer is not long enough")
                time.sleep(1)

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
                    newer.hist[new_] = older.hist[old_]
                    newer.lives[new_] = 4
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
                newer.hist.append(older.hist[old_])

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

        if buff_size == 1:
            return self.buffers[type_][0].id
        elif buff_size == 0:
            return False

        older = self.buffers[type_].popleft()
        newer = self.buffers[type_][0]
        [indices,newer] = self.match_ROIs(older,newer)
        self.buffers[type_][0] = newer

        return indices

    def draw_rects(self,img,rois):
        for lives,roi in zip(rois.lives,rois.roi):
            ''''ry1, rx1, ry2, rx2'''
            if(lives == 4):
                cv2.rectangle(img,(roi[1],roi[0]),(roi[3],roi[2]),(0,255,0),3)
        return img

    def construct_frame(self):
        w = int(self.output_width/2)

        raw = self.buffers["image"][0].frame.copy()
        #print("raw image shape is " + str(raw.shape))
        #print("int is " + str(raw.shape[1]/w))
        h = int(self.output_height/2)#int(raw.shape[0]/(raw.shape[1]/w))
        output = np.zeros((h * 2, w * 2, 3), dtype="uint8")

        if(len(self.buffers['camshift']) > 0):

            self.buffer_locks["mask"].acquire()
            mask = self.buffers["mask_image"][-1]
            #mask = self.draw_rects(raw,mask.roi)
            mask = imutils.resize(mask,width=w,height=h)
            self.buffer_locks["mask"].release()

            self.buffer_locks["camshift"].acquire()
            camshift = self.buffers['camshift'][-1]
            camshift = self.draw_rects(raw.copy(),camshift)
            camshift = imutils.resize(camshift,width=w,height=h)
            self.buffer_locks["camshift"].release()

            output[0:h, w:w * 2] = mask
            output[h:h * 2, w:w * 2] = camshift

        raw = imutils.resize(raw, width=w)
        output[0:h, 0:w] = raw
        return output

    def write_to_video(self):
        img = self.construct_frame()
        self.output_videos['combined'].write(img)
        #cv2.imshow("comb",img)
        name = str(self.count) + ".jpg"
        #cv2.imwrite(name,img)
        self.count += 1;
        #print("written_to_video")

    def mask_predict(self):

        while(self.stop == False):
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
                print("R shape" + str(r["features"].shape))
                print("ROIs shape " + str(r['rois'].shape))
                res_ = roi_class(r['rois'],image.time)

                self.buffer_locks["mask"].acquire()

                self.buffers['mask'].append(res_)


                #TRACK from LAST Mask position
                r['ids'] = self.track('mask')
                img = visualize.display_instances(image.frame, r['rois'], r['masks'],r['class_ids'], self.class_names, r['ids'], r['scores'])
                self.buffers["mask_image"].append(img)

                self.buffer_locks["camshift"].acquire()
                self.buffers["camshift"].append(self.buffers['mask'][-1])
                self.camshift_update = True
                self.buffer_locks["camshift"].release()

                self.buffer_locks["mask"].release()
                #print(results)
                end = time.time()
                print("Took around : " + str(end-start))
                #visualize.display_instances(image.frame, r['rois'], r['masks'],r['class_ids'], self.class_names, r['ids'], r['scores'])
            else:
                time.sleep(1)



    def callback(self,frame):#ros_data):
        frame_time = time.time()
        image = image_class(frame,frame_time)
        self.buffers["image"].append(image)
        time.sleep(0.05)
        self.write_to_video()

        #self.mask_predict()


def main(args):
    '''Initializes and cleanup ros node'''
    ic = model()

    mask_rcnn_thread = threading.Thread(target=ic.mask_predict)
    mask_rcnn_thread.daemon = True
    mask_rcnn_thread.start()

    '''
    camshift_thread = threading.Thread(target=ic.camshift)
    camshift_thread.daemon = True
    camshift_thread.start()
    '''
    print(os.getcwd())
    path = '/home/peter/catkin_ws/src/mask_rcnn/src/mask_rcnn/vid.mp4'
    cap = cv2.VideoCapture(0)
    if(cap.isOpened() == False):
      print("Failed to get camera")
    else:
        #rospy.init_node('mask_rcnn', anonymous=True)
        try:
            #rospy.spin()
            print("Located camera")
            # take first frame of the video
            while(1):
                ret,frame = cap.read()
                if ret ==True:
                    ic.callback(frame)
                else:
                    print("Failed")
                    break

        except KeyboardInterrupt:
            print("shutting down")
            ic.stop = True

        mask_rcnn_thread.join()
        #camshift_thread.join()
        print ("Shutting down ROS Image feature detector module")
        ic.output_writers["combined"].stop()
        ic.output_videos['combined'].release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
