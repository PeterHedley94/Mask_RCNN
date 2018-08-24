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

##File containing the main parameters used for the neural network


import os

#GANN MUST BE DIVISIBLE BY 4!!!
IM_HEIGHT = 200
IM_WIDTH = 200
RAW_HEIGHT = 1024
RAW_WIDTH = 1280

'''
IM_HEIGHT = 299
IM_WIDTH = 299'''

PRETRAINED_MODEL = False
NUMBER_CLASSES = 20
BATCH_SIZE = 16
NUMBER_EPOCHS = 500
NUMBER_CHANNELS = 3
IMAGES_PER_FOLDER = 12
SEND_TO_SLACK = False

CHECKPOINTS_FOLDER = os.path.join('MODEL_OUTPUTS','checkpoints')
MODEL_SAVE_FOLDER = os.path.join('MODEL_OUTPUTS','models')
OLD_MODELS_FOLDER = os.path.join('MODEL_OUTPUTS','old_models')
TENSORBOARD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','logs')
TENSORBOARD_OLD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','old_logs')
INTERMEDIATE_FILE = os.path.join('MODEL_OUTPUTS','checkpoints','intermediate.hdf5')
JSON_LOG_FILE = os.path.join('MODEL_OUTPUTS','loss_log.json')
JSON_OLD_LOGS_FOLDER = os.path.join('MODEL_OUTPUTS','old_json')

#THESE PARAMETERS BELOW ARE NOT USED IN MASTER SUBMITTED AS IT USES VGG_TESTING

SOURCE = os.path.join("DATA","product-image-dataset3")
TRAIN_DATA = os.path.join("DATA","f_d22_training_data")#"March-18","training_data")
VALIDATE_DATA = os.path.join("DATA","f_d22_validation_data")#"March-18","validation_data")
TEST_DATA = os.path.join("DATA","March-18","test_data")
DEBUG_FOLDER = os.path.join("DATA","DEBUGGING_DATA","debug_folder")


TRAIN_DATA_GROUPED = os.path.join("DATA","training_data_grouped")
VALIDATE_DATA_GROUPED = os.path.join("DATA","validation_data_grouped")
TEST_DATA_GROUPED = os.path.join("DATA","test_data_grouped")
