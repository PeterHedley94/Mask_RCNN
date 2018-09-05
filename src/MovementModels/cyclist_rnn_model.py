#!/usr/bin/env python
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import numpy as np,math

from keras.layers import Input, LSTM
from keras.models import Model
import keras
from keras import backend as K
 #set learning phase
K.set_learning_phase(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.models import load_model
from keras.layers import TimeDistributed
from utils.common import *


class rcnn_total_model:
    def __init__(self,complete_cached_model):


        #Inputs
        pose_input = Input(shape=(sequence_length,pose_data_dims[0]*pose_data_dims[1]))
        video_input = Input(shape=(sequence_length, IM_HEIGHT_model, IM_WIDTH_model, NUMBER_CHANNELS))


        #IMAGES RCNN MODE
        self.cnnmodel = self.get_cnn_model(conv1_size,conv2_size,no_layers)
        encoded_frame_sequence = TimeDistributed(self.cnnmodel)(video_input) # the output will be a sequence of vectors
        encoded_video_1 = LSTM(64)(encoded_frame_sequence)
        self.encoded_video = Dense(32)
        encoded_video = self.encoded_video(encoded_video_1)


        #POSE SEQUENCE MODEL
        self.posemodel = self.get_pose_model()
        pose_model_branch = self.posemodel(pose_input)

        #Total MODEL
        output_branches = keras.layers.Concatenate(axis=-1)([encoded_video,pose_model_branch])
        final_dense = Dense(64, activation='linear')(output_branches)
        total_output= Dense(number_outputs, activation='linear')(final_dense)
        self.total_model = Model(inputs=[video_input,pose_input], outputs=total_output)

        self.total_model = load_model(complete_cached_model)

        sgd = SGD(lr=10**(-3.5), decay=10**(-6), momentum=0.9, nesterov=True)
        self.total_model.compile(loss='mse', optimizer=sgd)



    def get_cnn_model(self,conv1_size,conv2_size,no_layers):
        cnnmodel = Sequential()
        cnnmodel.add(Conv2D(20, (3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS)))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmodel.add(Dropout(0.25))
        cnnmodel.add(Conv2D(32, (3, 3), activation='relu'))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmodel.add(Dropout(0.25))
        cnnmodel.add(Flatten())
        return cnnmodel

    def get_pose_model(self):
        # expected input data shape: (batch_size, timesteps, data_dim)
        model = Sequential()
        model.add(LSTM(63, return_sequences=True,input_shape=(sequence_length,pose_data_dims[0]*pose_data_dims[1])))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(53, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
        return model

    def predict(self,x,x_images):
        return self.total_model.predict(x_images,x)
