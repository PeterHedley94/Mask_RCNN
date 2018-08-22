import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np,random

from MovementModels.data_generator import *

dir = "/home/peter/Documents/okvis_drl/build/bedroom_dataset"

params_train = {'dir': dir,
              'batch_size': 36,
              'shuffle': True,
                'sequence_length': 12,'time_distributed' : True}

train_generator = DataGenerator(**params_train)
train_gen = train_generator.generate()
