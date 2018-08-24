#!/usr/bin/env python
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from keras.layers import Input, LSTM
from keras.models import Model
from MovementModels.data_generator import *
import keras
from keras import backend as K
 #set learning phase
K.set_learning_phase(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,Adam
from keras.models import load_model
from MovementModels.callbacks import *
from utils.folder_manipulation import *
from MovementModels.common_network_operations import *
from keras.layers import TimeDistributed

class CNN_LSTM(object):
    def __init__(self,output = True,lr=0.001,cached_model= None):
        self.model_name = "vgg_net"
        self.output = output
        number_outputs = 3
        self.data_dims = [3,4] #3x4 camera matrix
        self.sequence_length_ = 4
        self.batch_size = 2

        images = True
        self.image_height = 480#180
        self.image_width= 640#240
        self.no_image_channels=3
        self.camera_model_location = os.path.join("utils",'test_camera_model.json')

        #Basic working model
        self.model = Sequential()
        self.model.add(LSTM(32,return_sequences=True,activation = "linear",input_shape=(self.sequence_length_,self.data_dims[0]*self.data_dims[1])))  # returns a sequence of vectors of dimension 32
        self.model.add(LSTM(32,activation = "linear"))
        self.model.add(Dense(number_outputs,activation = "linear"))

        lr = 0.00001
        sgd = SGD(lr=lr, decay=5e-5, momentum=0.9, nesterov=True)
        rms = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
        adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mse', optimizer=sgd)

        '''
        self.cnnmodel = self.get_cnn_model()
        video_input = Input(shape=(self.sequence_length, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS))
        encoded_frame_sequence = TimeDistributed(self.cnnmodel)(video_input) # the output will be a sequence of vectors
        encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be one vector
        '''

        '''
        self.model = Sequential()
        self.model.add(LSTM(32, return_sequences=True,
                       input_shape=(self.sequence_length_, self.data_dims)))  # returns a sequence of vectors of dimension 32
        #self.model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        self.model.add(LSTM(32))  # return a single vector of dimension 32
        self.model.add(Dense(number_outputs))'''

        '''
        #BEST SO FAR!!-model2
        batch_size = 18
        timesteps = self.sequence_length_
        data_dim = 13

        self.model = Sequential()
        self.model.add(LSTM(32, return_sequences=True, stateful=True,
                       batch_input_shape=(self.batch_size, timesteps, data_dim)))
        self.model.add(LSTM(32, return_sequences=True, stateful=True))
        self.model.add(LSTM(32, stateful=True))
        self.model.add(Dense(3, activation='linear'))
        #self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])'''


        #model-3
        '''
        pose_input = Input(shape=(self.sequence_length_,self.data_dims))
        self.model = Sequential()
        self.model.add(Dense(32,activation='linear',input_shape=(self.sequence_length_,self.data_dims)))
        self.model.add(Dense(32,activation='linear'))
        encoded_sequence = TimeDistributed(self.model)(pose_input)
        total_sequence = LSTM(32)(encoded_sequence)
        pose_model_output = Dense(3, activation='linear')(total_sequence)
        output = Dense(number_outputs, activation='linear')(pose_model_output)#([encoded_video,self.imumodel])
        self.model = Model(inputs=pose_input, outputs=output)

        sgd = SGD(lr=lr, decay=5e-5, momentum=0.9, nesterov=True)
        rms = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
        adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mse', optimizer=sgd)'''

        '''
        pose_input = Input(shape=(self.sequence_length_,self.data_dims))
        self.posemodel = self.get_pose_model()
        pose_model_output = self.posemodel(pose_input)

        output = Dense(number_outputs, activation='linear')(pose_model_output)#([encoded_video,self.imumodel])
        self.model = Model(inputs=pose_input, outputs=output)

        if cached_model is not None:
            self.model = load_model(cached_model)

        sgd = SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])'''

    def get_pose_model(self):
        # expected input data shape: (batch_size, timesteps, data_dim)
        model = Sequential()
        model.add(LSTM(32, return_sequences=True,
                       input_shape=(self.sequence_length_, self.data_dims)))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
        model.add(LSTM(32))  # return a single vector of dimension 32
        return model

    def get_cnn_model(self):
        cnnmodel = Sequential()
        cnnmodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS)))
        cnnmodel.add(Conv2D(32, (3, 3), activation='relu'))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmodel.add(Dropout(0.25))
        cnnmodel.add(Conv2D(64, (3, 3), activation='relu'))
        cnnmodel.add(Conv2D(64, (3, 3), activation='relu'))
        cnnmodel.add(MaxPooling2D(pool_size=(2, 2)))
        cnnmodel.add(Dropout(0.25))
        cnnmodel.add(Flatten())
        return cnnmodel

    #CHANGE BATCH SIZE BACK!!
    def train(self,train_directory_, validation_directory_,model_description,epochs):
        self.model_name += model_description
        create_folder_structure()

        params_val = {'dir': validation_directory_,
                  'batch_size': self.batch_size,
                  'shuffle': True,'debug_mode':False,
                  'sequence_length' : self.sequence_length_ ,'time_distributed' : True,'images':False,
                  'image_height':self.image_height,'image_width':self.image_width,
                  'no_image_channels':self.no_image_channels,'camera_model_location':self.camera_model_location}

        validation_generator = DataGenerator(**params_val)
        validate_gen = validation_generator.generate()

        params_train = {'dir': train_directory_,
                  'batch_size': self.batch_size,
                  'shuffle': True,'debug_mode':False,
                  'sequence_length' : self.sequence_length_ ,'time_distributed' : True,'images':False,
                  'image_height':self.image_height,'image_width':self.image_width,
                  'no_image_channels':self.no_image_channels,'camera_model_location':self.camera_model_location}

        train_generator = DataGenerator(**params_train)
        train_gen = train_generator.generate()

        #CHECKS!################
        test_in = train_gen.__next__()
        test_in_val = validate_gen.__next__()

        steps_per_epoch_ =  train_generator.batches_per_epoch
        validation_steps_ = validation_generator.batches_per_epoch
        ##########################

        calls_ = logs()
        '''
        self.model.fit_generator(train_gen, validation_data=validate_gen,
                                 callbacks=[calls_.json_logging_callback,
                                            calls_.slack_callback,
                                            get_model_checkpoint(),get_Tensorboard()], steps_per_epoch =steps_per_epoch_,
                                                                                    validation_steps=steps_per_epoch_, epochs=epochs)'''

        self.model.fit_generator(train_gen, validation_data=validate_gen, steps_per_epoch =steps_per_epoch_,
                                                                                    validation_steps=steps_per_epoch_, epochs=epochs)

        current_directory = os.path.dirname(os.path.abspath(__file__))
        print("Model saved to " + os.path.join(current_directory, os.path.pardir,MODEL_SAVE_FOLDER,self.model_name + '.hdf5'))
        if not os.path.exists(MODEL_SAVE_FOLDER):
            os.makedirs(MODEL_SAVE_FOLDER)
        self.model.save(os.path.join(MODEL_SAVE_FOLDER,str(self.model_name + '.hdf5')))
        clean_up(self.model_name)


    def predict(self,input_data):
        K.set_learning_phase(0)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
        # CHANGED THIS!!!!
        input_data = input_data / 255
        predictions = self.model.predict(input_data, verbose=False)
        return np.array(predictions[0])


    def return_weights(self,layer):
        return self.model.layers[layer].get_weights()
