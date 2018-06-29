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
IMAGE_COUNT = 0
#RUNNING WITH ROS
#ROOT_DIR = os.path.join(os.path.abspath("./"),"src","mask_rcnn","src","mask_rcnn")
VERBOSE=False
#image_buffer
from utils.image_class import *
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
    def __init__(self,im_width,im_height):

        #Create buffers for threads
        mask_roi_buffer_lock = threading.Lock()
        self.buffers = {"mask":collections.deque(maxlen=2),"mask_image":collections.deque(maxlen=2),"image":collections.deque(maxlen=2),"camshift":collections.deque(maxlen=2)}
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

        self.initialise_ros()
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


    def initialise_ros(self):
        '''Initialize ros publisher, ros subscriber'''
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera/image_raw",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")


    def camshift(self):
        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 1 )
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
                        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
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
                if(matches[old_,new_]> self.IOU_threshold):
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
            if (max_val < self.IOU_threshold and older.lives[old_] > 0):
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

    def max_index_selector(self,array):
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


    def match_feature_similarity(self,older,newer):

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

        print("Older id is " + str(older.id))
        print("Older features shape " + str(older.features.shape))
        similarity = np.zeros((max(new_id)+1,max(old_id)+1))


        for old_roi in range(len(old_id)):
            for new_roi in range(len(new_id)):
                similarity[new_id[new_roi],old_id[old_roi]] = np.absolute(np.mean(new_features[new_roi,:,:,:]-old_features[old_roi,:,:,:]))
        #newer.id = np.argmin(similarity[1:,1:],axis = 1)+1

        newer.id = self.max_index_selector(similarity[1:,1:,None])
        print("Newer ids are " + str(newer.id))
        print("Older Colours are " + str(older.colours))
        for ind in newer.id:
            print("Ind is " + str(ind))
            print("Len older is is " + str(len(older.id)))
            if ind < len(older.id)-1:
                newer.colours[ind-1] = older.colours[ind-1]
            else:
                continue
        #newer.colours = [older.colours[ind-1] for ind in newer.id if ind<len(older.id)]
        print("Newer Colours are : " + str(newer.colours))
        print(similarity)
        newer.lives = [x+1  if x<7 else x for x in newer.lives]
        print("Newer id pt 2 is " + str(newer.id))
        if(newer.features.shape[0] != len(newer.id)):
            print("########################################")
        return newer.id,newer


    def track(self,type_):

        buff_size = len(self.buffers[type_])

        if buff_size == 1:
            return self.buffers[type_][0].id
        elif buff_size == 0:
            return False

        older = self.buffers[type_][-2]
        newer = self.buffers[type_][-1]
        [indices,newer] = self.match_ROIs(older,newer)
        #[indices,newer] = self.match_feature_similarity(older,newer)
        #newer.id = np.arange(len(newer.id)) +1
        self.buffers[type_][0] = newer

        return indices

    def draw_rects(self,img,rois):
        global IMAGE_COUNT
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale = 1
        fontColor=(255,255,255)
        lineType=2

        for lives,roi,id,class_,colour in zip(rois.lives,rois.roi,rois.id,rois.class_,rois.colours):
            ''''ry1, rx1, ry2, rx2'''
            if(lives > -100 and class_ in [1,2,3,4,5,6,7,8]):
                pos_ = (roi[1]+2,roi[0]+10)
                cv2.rectangle(img,(roi[1],roi[0]),(roi[3],roi[2]),colour,3)
                image_name = os.path.join("output_images",str(IMAGE_COUNT) + ".jpg")
                cv2.imwrite(image_name,img[roi[0]:roi[2],roi[1]:roi[3]])
                IMAGE_COUNT += 1
                text_ = self.class_names[class_] + " : " + str(id)
                cv2.putText(img,text_, pos_, font,
                fontScale, fontColor, lineType)

    def draw_masks(self,img,rois):
        for lives,mask,colour,class_ in zip(rois.lives,rois.masks,rois.colours,rois.class_):
            if(lives > 4 and class_ in [1,2,3,4,5,6,7,8]):
                img = visualize.apply_mask(img, mask, colour)
        return img

    def construct_frame(self):
        raw = self.buffers["image"][0].frame.copy()
        self.buffer_locks["mask"].acquire()

        mask = raw.copy()
        self.draw_rects(mask,self.buffers['mask'][-1])
        mask = visualize.display_instances(mask, self.buffers['mask'][-1].masks,colors=self.buffers['mask'][-1].colours)
        mask = imutils.resize(mask,width=self.output_width,height=self.output_height)
        self.buffer_locks["mask"].release()
        return mask

    def write_to_video(self):
        img = self.construct_frame()
        self.output_videos['combined'].write(img)
        name = str(self.count) + ".jpg"
        cv2.imwrite("image.jpg",img)
        self.count += 1;

    def mask_predict(self):
        try:
            image = self.buffers["image"][0]
        except IndexError:
            image = False
        if(image != False):
            start = time.time()
            with graph.as_default():
                #results contains ['rois', 'scores', 'class_ids', 'masks']
                results = self.model.detect([image.frame], verbose=1)

            # Put resuts in buffer
            r = results[0]
            print("R masks shape is :" + str(r['masks'].shape))
            print("R rois shape is : " + str(r['rois'].shape))
            print("Features shape is :" + str(r['features'].shape))
            res_ = roi_class(r['rois'],image.time,r['class_ids'],
            visualize.random_colors(r['rois'].shape[0]),r['masks'],r['features'])
            #print("Masks Shape is: " + str(len(res_.masks)))
            self.buffer_locks["mask"].acquire()
            self.buffers['mask'].append(res_)

            #TRACK from LAST Mask position
            r['ids'] = self.track('mask')
            self.buffer_locks["mask"].release()
            #print(results)
            end = time.time()
            print("Took around : " + str(end-start))


    def callback(self,frame):#ros_data):
        frame_time = time.time()
        image = image_class(frame,frame_time)
        self.buffers["image"].append(image)
        self.mask_predict()
        self.write_to_video()

        #self.mask_predict()

def print_progress(counter,total):
    inc = 5 #print every 10%
    percentage = str(counter/total * 100)
    title_string = percentage + str('% [')
    for i in range(round(counter/total * 100/inc)):
        title_string = title_string + '=='
    title_string = title_string + '>'
    for i in range(round(100/inc-round(counter/total * 100/inc))):
        title_string = title_string + '__'
    title_string = title_string + ']'
    print(title_string)

def main(args):
    '''Initializes and cleanup ros node'''
    path = '/home/peter/catkin_ws/src/mask_rcnn/src/mask_rcnn/gold.avi'
    cap = cv2.VideoCapture(path)
    frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
    ic = model(frame_width,frame_height)

    print(os.getcwd())
    frame_no = 0

    if(cap.isOpened() == False):
      print("Failed to get camera")
    else:
        length_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Found " + str(length_video) + " images")
        #rospy.init_node('mask_rcnn', anonymous=True)
        try:
            #rospy.spin()
            print("Located camera")
            # take first frame of the video
            while(1):
                ret,frame = cap.read()
                if ret ==True:
                    if frame_no > 900:
                        ic.callback(frame)
                    else:
                        print_progress(frame_no,length_video)
                        frame_no += 1
                else:
                    print("Failed")
                    break

        except KeyboardInterrupt:
            print("shutting down")
            ic.stop = True

        print ("Shutting down ROS Image feature detector module")
        ic.output_videos['combined'].release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
