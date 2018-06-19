import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')

import cv2,imutils

'''
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
'''
import numpy as np
import cv2

path = '/home/peter/catkin_ws/src/mask_rcnn/src/mask_rcnn'
cap = cv2.VideoCapture('/home/peter/catkin_ws/src/mask_rcnn/src/mask_rcnn/vid.mp4')

if(cap.isOpened() == False):
    print("Video not located")

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


'''
cap = cv2.VideoCapture('/mask_rcnn/vid.mp4')
output_writers = {"combined":cv2.VideoWriter_fourcc(*'XVID')}
output_videos = {"combined":cv2.VideoWriter('combined.avi',output_writers["combined"], 20.0, (640,480))}

count = 0
if(cap.open(0) == False):
  print("Failed to get camera")
else:
    while (count < 100):
        ret,frame = cap.read()
        if ret ==True:
            output_videos['combined'].write(frame)
            name = str(count) + ".jpg"
            cv2.imwrite(name,frame)
        else:
            print("Failed")
            break
        count += 1


#output_writers["combined"].stop()
output_videos['combined'].release()
'''
