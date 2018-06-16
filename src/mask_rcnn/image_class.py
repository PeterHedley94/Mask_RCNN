import collections
import numpy as np

class image_class:
    def __init__(self,frame_,time_):
        self.time = time_
        self.frame = frame_

class circular_buffer:
    #Creating a circular buffer for pictures
    def __init__(self):
        self.buffer = collections.deque(maxlen=2)

    def insert(self,image):
        self.buffer.append(image)
        #print("Buffer size is now " + str(len(self.image_buffer)))

    def remove(self):
        try:
            return self.buffer.pop()
        except IndexError:
            return False

class roi_class:
    def __init__(self,roi_,time_):
        self.time = time_
        self.roi = roi_
        self.id = np.arange(roi_.shape[0]) + 1
        self.lives = [4] * roi_.shape[0]
