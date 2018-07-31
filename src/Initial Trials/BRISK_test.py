import os,sys,inspect
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
import numpy as np
import cv2
from matplotlib import pyplot as plt
import shutil

import numpy as np
import cv2
from matplotlib import pyplot as plt


class BRISK_class:
    def __init__(self):
        # Initiate BRISK detector
        self.brisk = cv2.BRISK_create()
        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def get_des_key(self,img,rois):
        des = []
        key = []
        for roi in rois:
            key1,des1 = self.brisk.detectAndCompute(img[roi[0]:roi[2],roi[1]:roi[3]],None)
            des.append(des1)
            key.append(key1)
        return des,key

    def get_matches(self,des1,des2):
        return self.bf.match(des1,des2)

    def get_sum_matches(self,older,old_,newer,new_):
        sum = 0
        matches = self.get_matches(older.descriptors[old_],newer.descriptors[new_])
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        if len(matches) < 10:
            for i in matches:
                sum += i.distance
        else:
            for i in matches[:10]:
                sum += i.distance
        return sum

#RETURNS ALL IMAGES NAMES IN A DIRECTORY
def get_image_names(directory_):
    files = os.listdir(directory_)
    image_present = False
    images = []
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            images.append(file)
            image_present = True

    if not image_present:
        print("Could not find any Images!")
    return images

dir = "DATA/Test_images"
print(os.path.join(dir,'0.jpg'))
print(os.getcwd())
print(cv2.imread(os.path.join(dir,'0.jpg'),0))
classes = []
classes.append(cv2.imread(os.path.join(dir,'0.jpg'),0))
classes.append(cv2.imread(os.path.join(dir,'1.jpg'),0))
classes.append(cv2.imread(os.path.join(dir,'2.jpg'),0))
print(classes)

# Initiate BRISK detector
brisk = cv2.BRISK_create()
# find the keypoints and descriptors with ORB
descriptors = []
keypoints = []
for class_ in classes:
    #cv2.imshow("fff",class_)
    #cv2.waitKey(10)
    kp, des = brisk.detectAndCompute(class_,None)
    descriptors.append(des)
    keypoints.append(kp)

image_names = get_image_names(dir)
print(image_names)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


for img in image_names:
    best_match = 0
    best_value = 10000000#0
    img_ = cv2.imread(os.path.join(dir,img),0)
    kp, des = brisk.detectAndCompute(img_,None)
    for class_ in range(len(classes)):
        # Match descriptors.
        #print(des)
        matches = bf.match(des,descriptors[class_])
        #matches = bf.knnMatch(des,descriptors[class_], k=10)
        sum = 0

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        if len(matches) < 10:
            for i in matches:
                sum += i.distance
        else:
            for i in matches[:10]:
                sum += i.distance

        print("Image " + str(img) + " vs class : " + str(class_) + " = " + str(sum) + " or :" + str(len(matches)))
        if sum < best_value:
            best_value = sum
            best_match = class_

    shutil.move(os.path.join(dir,img),os.path.join(dir,str(best_match),img))
    '''
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(classes[class_],keypoints[class_],img_,kp,matches[:10],img_, flags=2)
    cv2.imshow("frame img3",classes[class_])
    cv2.waitKey(10)

    print("Image " + str(img) + " matches class " + str(class_))

    '''
'''
# Match descriptors.
#matches = bf.match(des1,des2)
print(matches[0])
for i in matches:
    print(i.distance)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],img2, flags=2)
plt.imshow(img3),plt.show()'''
