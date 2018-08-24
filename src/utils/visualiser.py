#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import cv2, numpy as np,imutils, random,math
from utils.pose_visualiser import *
from utils.object_map_visualiser import *
IMAGE_COUNT = 0
ROI_IMAGE_COUNT = 0


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image, masks,colors=None,show_mask=True,
                       captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """

    # If no axis is passed, create one and automatically call show()
    N = len(colors)
    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[:,:,i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

    return masked_image.astype(np.uint8)


def check_in_frame(pt):
    x = pt[0]
    y = pt[1]
    if x<0 or x>680:
        return False
    if y<0 or y>480:
        return False
    return True

def x_y_from_cx_cy(array):
    centre_x,centre_y,width,height = array
    pt1 = (int(centre_x-width/2),int(centre_y-height/2))
    pt2 = (int(centre_x+width/2),int(centre_y+height/2))
    return pt1,pt2


def draw_rects(img,m_,class_names):
    global ROI_IMAGE_COUNT
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale = 0.2
    fontColor=(255,255,255)
    lineType=2

    for i in range(m_.no_rois):
        #lives,roi,id,class_,colour,key1,kalman
        #,mrcnn_out.roi_dims_c,mrcnn_out.id,mrcnn_out.colours,mrcnn_out.keypoints,mrcnn_out.kalman
        ''''ry1, rx1, ry2, rx2'''
        if m_.lives[i] > -100:# and m_.class_[i] in [1,2,3,4,5,6,7,8]: #[40]
            pt1,pt2 = x_y_from_cx_cy(m_.roi_dims_c[[0,1,4,5],i])
            if check_in_frame(pt1) and check_in_frame(pt2):
                pos_ = np.array(pt1)
                pos_ = (pos_[0]+2,pos_[1]+10)
                cv2.rectangle(img,pt1,pt2,(0,0,0))
                cv2.putText(img,str(m_.id[i]), pos_, font,fontScale, fontColor, thickness = 1)

            #print("The real state is " + str(m_.roi_dims_w[[0,1,4,5],i]))
            #print("The kalman predicted state is" + str(m_.kalman[i].statePre))
            k_c_x,k_c_y = m_.world_to_camera(m_.kalman[i].statePre[:3])[:2]#.append(kalman.statePre[[4,5]])
            k_w,k_h = m_.kalman[i].statePre[[3,4]]
            print("Kalman predicted state is  \n" + str(m_.kalman[i].statePre))
            print("Kalman Q state is  \n" + str(m_.kalman[i].Q))
            print("Kalman R state is  \n" + str(m_.kalman[i].R))
            print("Kalman predicted in camera frame is " + str(m_.world_to_camera(m_.kalman[i].statePre[:3])))
            pt1,pt2 = x_y_from_cx_cy([k_c_x,k_c_y,k_w,k_h])
            if check_in_frame(pt1) and check_in_frame(pt2):
                pos_ = np.array(pt1)
                pos_ = (pos_[0]+2,pos_[1]+10)
                cv2.rectangle(img,pt1,pt2,(0,255,0))
                cv2.putText(img,str(m_.id[i]), pos_, font,fontScale, (0,255,0), thickness = 1)

def draw_depth_text(img,m_):
    global ROI_IMAGE_COUNT
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale = 0.8
    fontColor=(0,0,255)
    lineType=2

    for i in range(m_.no_rois):
        #lives,roi,id,class_,colour,key1,kalman
        #,mrcnn_out.roi_dims_c,mrcnn_out.id,mrcnn_out.colours,mrcnn_out.keypoints,mrcnn_out.kalman
        ''''ry1, rx1, ry2, rx2'''
        if m_.lives[i] > -100:# and m_.class_[i] in [1,2,3,4,5,6,7,8]:
            pt = tuple([int(m_.roi_dims_c[[0],i]),int(m_.roi_dims_c[[1],i])])
            if check_in_frame(pt):
                cv2.putText(img,str(round(m_.depth_rois[i],2)), pt, font,fontScale, fontColor, thickness = 3)


def draw_masks(img,mrcnn_out):
    for lives,mask,colour,class_ in zip(mrcnn_out.lives,mrcnn_out.masks,mrcnn_out.colours,mrcnn_out.class_):
        if(lives > 0 and class_ in [1,2,3,4,5,6,7,8]):
            img = apply_mask(img, mask, colour)
    return img


def get_depth_plot(array):
    h,w = array.shape
    array2 = np.zeros((h,w))
    cv2.normalize(array,array2,0,255,cv2.NORM_MINMAX)
    nice_array = np.zeros((h,w,3))
    for channel in range(3):
        nice_array[:,:,channel] = array
    return nice_array


class visualiser:
    def __init__(self,w,h):
        self.w = w
        self.h = h
        self.pv = pose_visualiser(int(w),int(h*2/3))
        self.ov = obj_map_visualiser(int(w),int(h*2/3))

    def construct_frame(self,mrcnn_output,class_names,T_WS_r,T_WS_C,camera_model,cycle_model):
        global IMAGE_COUNT,pv
        raw = mrcnn_output.image.copy()
        w,h = int(self.w/2), int(self.h/3)

        #MASKS
        mask_image = raw.copy()
        mask_image = display_instances(mask_image, mrcnn_output.masks,colors=mrcnn_output.colours)
        draw_rects(mask_image,mrcnn_output,class_names)
        #cv2.imwrite(str(IMAGE_COUNT) + ".jpg",mask_image)
        cv2.imwrite("progress.jpg",mask_image)
        IMAGE_COUNT += 1
        mask_image = imutils.resize(mask_image,width=w,height=h)

        #POSE
        self.pv.add_points(T_WS_r)
        pose_image = self.pv.plot(mrcnn_output.roi_dims_w[:2,:],cycle_model)
        self.ov.set_limits(self.pv.xlims,self.pv.ylims)
        pose_image = self.ov.plot(pose_image,mrcnn_output,class_names)


        #DEPTH IMAGE
        depth_image = get_depth_plot(mrcnn_output.depth.copy())
        cv2.imwrite(os.path.join("depth",str(IMAGE_COUNT)+".jpg"),depth_image)
        depth_image = display_instances(depth_image, mrcnn_output.masks,colors=mrcnn_output.colours)
        draw_depth_text(depth_image,mrcnn_output)

        #COMBINED
        total_image = np.zeros((self.h,self.w,3),dtype=np.uint8)
        total_image[0:h,0:w,:] = mask_image
        total_image[0:h,w:,:] = imutils.resize(depth_image,width=w,height=h)
        total_image[h:,:,:] = pose_image
        print(total_image.shape)
        return total_image


    def write_to_video(self,output_video,mcrnn_output,class_names,T_WS_r,T_WS_C,camera_model,cycle_model):
        img = self.construct_frame(mcrnn_output,class_names,T_WS_r,T_WS_C,camera_model,cycle_model)
        cv2.imshow("i",img)
        cv2.waitKey(1)
        print(img.shape)
        cv2.imwrite("image.jpg",img)
        output_video.write(img)
