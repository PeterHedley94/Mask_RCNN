import sys,os
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

sct = mss()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid = cv2.VideoWriter('output.avi', fourcc, 20.0, (mon['width'],mon['height']))

while 1:
    img = sct.grab(mon)
    #qqimg = Image.frombytes('RGB', (mon['width'],mon['height']), sct.image)
    img_np = np.array(img)
    print(img_np.shape)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
    print(frame.shape)
    vid.write(frame)
    #cv2.imshow('image',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break
