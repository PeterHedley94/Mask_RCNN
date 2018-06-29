import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
import cv2
from matplotlib import pyplot as plt
import shutil

import numpy as np
import cv2
from matplotlib import pyplot as plt

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

'''
img = cv2.imread('Test_images/0.jpg',0)

plt.imshow(img),plt.show()
# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img3 = img.copy()
img2 = cv2.drawKeypoints(img,kp,img3,color=(0,255,0), flags=0)
plt.imshow(img2),plt.show()

'''
print(os.path.join('Test_images','0.jpg'))
print(os.getcwd())
print(cv2.imread(os.path.join('Test_images','0.jpg'),0))
classes = []
classes.append(cv2.imread(os.path.join('Test_images','0.jpg'),0))
classes.append(cv2.imread(os.path.join('Test_images','1.jpg'),0))
classes.append(cv2.imread(os.path.join('Test_images','2.jpg'),0))
print(classes)
'''
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
'''

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

dir = "Test_images"
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
        matches = bf.match(des,descriptors[class_])
        sum = 0

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        for i in matches[:10]:
            sum += i.distance

        print("Image " + str(img) + " vs class : " + str(class_) + " = " + str(sum) + " or :" + str(len(matches)))
        if sum < best_value: #len(matches) > best_value:
            best_value = sum#len(matches)
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
