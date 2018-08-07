#!/usr/bin/env python
import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import cv2, numpy as np,imutils, random,math
from utils.pose_visualiser import *
IMAGE_COUNT = 0
ROI_IMAGE_COUNT = 0


pv = pose_visualiser()

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

def draw_rectangle(img,centre_x,centre_y,width,height, color):
    pt1 = (int(centre_x-width/2),int(centre_y-height/2))
    pt2 = (int(centre_x+width/2),int(centre_y+height/2))
    cv2.rectangle(img,pt1,pt2,color)

def draw_rects(img,rois,class_names):
    global ROI_IMAGE_COUNT
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale = 0.2
    fontColor=(255,255,255)
    lineType=2

    for lives,roi,id,class_,colour,key1,kalman in zip(rois.lives,rois.roi,rois.id,rois.class_,rois.colours,rois.keypoints,rois.kalman):
        ''''ry1, rx1, ry2, rx2'''
        if(lives > -100 and class_ in [1,2,3,4,5,6,7,8]):
            pos_ = (roi[1]+2,roi[0]+10)
            cv2.drawKeypoints(img[roi[0]:roi[2],roi[1]:roi[3]], key1,img[roi[0]:roi[2],roi[1]:roi[3]],(0,255,255))
            cv2.rectangle(img,(roi[1],roi[0]),(roi[3],roi[2]),colour)
            state = kalman.statePre
            pos_2 = (int(state[0]-state[2]/2)+2,int(state[1]-state[3]/2)+10)
            draw_rectangle(img,state[0],state[1],state[2],state[3],(0,255,0))
            text_ = str(id)#class_names[class_] + " : " + str(id)
            cv2.putText(img,text_, pos_2, font,fontScale, (0,255,0), thickness = 1)
            #image_name = os.path.join("output_rois",str(ROI_IMAGE_COUNT) + ".jpg")
            #cv2.imwrite(image_name,img[roi[0]:roi[2],roi[1]:roi[3]])
            ROI_IMAGE_COUNT += 1
            cv2.putText(img,text_, pos_, font,
            fontScale, fontColor, thickness = 1)

def draw_masks(img,rois):
    for lives,mask,colour,class_ in zip(rois.lives,rois.masks,rois.colours,rois.class_):
        if(lives > 0 and class_ in [1,2,3,4,5,6,7,8]):
            img = apply_mask(img, mask, colour)
    return img


def construct_frame(raw_image,mrcnn_output,class_names,dims,T_WS_r,T_WS_C,camera_model):
    global IMAGE_COUNT,pv
    raw = raw_image.frame.copy()
    w,h = math.floor(dims[0]/2),math.floor(dims[1]/2)

    #MASKS
    mask_image = raw.copy()
    mask_image = display_instances(mask_image, mrcnn_output.masks,colors=mrcnn_output.colours)
    draw_rects(mask_image,mrcnn_output,class_names)
    cv2.imwrite("progress.jpg",mask_image)
    mask_image = imutils.resize(mask_image,width=w,height=h)

    #POSE
    pose_image = pv.pose_callback(raw_image.depth.copy(),T_WS_r,T_WS_C,camera_model,w*2,h*2)

    '''
    #COMBINED
    total_image = np.zeros((dims[1],dims[0],3),dtype=np.uint8)
    total_image[0:h,0:w,:] = mask_image
    total_image[0:h,w:,:] = imutils.resize(raw_image.depth.copy(),width=w,height=h)
    total_image[h:,0:w,:] = pose_image
    print(total_image.shape)'''
    return pose_image


def write_to_video(output_video,image,mcrnn_output,class_names,dims,T_WS_r,T_WS_C,camera_model):
    img = construct_frame(image,mcrnn_output,class_names,dims,T_WS_r,T_WS_C,camera_model)
    cv2.imshow("i",img)
    cv2.waitKey(10)
    print(img.shape)
    cv2.imwrite("image.jpg",img)
    output_video.write(img)
