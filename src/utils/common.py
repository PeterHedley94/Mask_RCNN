import os,sys
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')



BRISK_THRESHOLD = 40
BRISK_MIN_VALUE = 10**-3

KALMAN_LIKELIHOOD_THRESHOLD = 10**7
KALMAN_SURE_THRESHOLD = 10**6

#If Brisk finds no matches but Kalman is very sure then accept match
INDEX_SELECTOR_THRESHOLD = 10**-20#5000#(KALMAN_LIKELIHOOD_THRESHOLD-KALMAN_SURE_THRESHOLD) *BRISK_MIN_VALUE


expected_heights = {'person':[1.1,1.5,2.15],'bicycle':[0.6,1.0,1.2],"car":[1.2,1.6,2.5],'motorcycle':[0.6,1.0,1.2],
                    'bus':[2.5,3.0,3.5],'truck':[2.5,3.0,3.5],'bottle':[0.10,0.2,0.35]}
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
