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
from data_generator import *
import keras
from keras import backend as K
 #set learning phase
K.set_learning_phase(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from src.callbacks import *
from src.DATA_PREPARATION.folder_manipulation import *
from src.NN_MODELS.common_network_operations import *
from keras.layers import TimeDistributed

class CNN_LSTM(object):
    def __init__(self,output = True,lr=0.01,cached_model= None):
        self.sequence_length = 5
        self.model_name = "vgg_net"
        self.output = output

        self.cnnmodel = self.get_cnn_model()
        video_input = Input(shape=(self.sequence_length, IM_HEIGHT, IM_WIDTH, NUMBER_CHANNELS))
        encoded_frame_sequence = TimeDistributed(self.cnnmodel)(video_input) # the output will be a sequence of vectors
        encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be one vector

        imu_input = Input(shape=(self.sequence_length,imu_data_dims))
        self.imumodel = self.get_imu_model()

        output = Dense(NUMBER_CLASSES, activation='softmax')(self.imumodel)#([encoded_video,self.imumodel])
        self.model = Model(inputs=imu_input, outputs=output)

        if cached_model is not None:
            self.model = load_model(cached_model)

        sgd = SGD(lr, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def get_imu_model(self):
        # expected input data shape: (batch_size, timesteps, data_dim)
        model = Sequential()
        model.add(LSTM(32, return_sequences=True,
                       input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
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

    def train(self,train_directory_, validation_directory_,model_description,epochs):
        self.model_name += model_description
        create_folder_structure()

        params_val = {'dir': validation_directory_,
                  'batch_size': 36,
                  'shuffle': True,
                  'sequence_length' : 12,'time_distributed' : True}

        validation_generator = DataGenerator(**params_val)
        validate_gen = validation_generator.generate()

        params_train = {'dir': train_directory_,
                      'batch_size': 36,
                      'shuffle': True,
                        'sequence_length': 12,'time_distributed' : True}

        train_generator = DataGenerator(**params_train)
        train_gen = train_generator.generate()

        #CHECKS!################
        test_in = train_gen.__next__()
        test_in_val = validate_gen.__next__()

        steps_per_epoch_ =  train_generator.batches_per_epoch
        validation_steps_ = validation_generator.batches_per_epoch
        ##########################

        calls_ = logs()
        self.model.fit_generator(train_gen, validation_data=validate_gen,
                                 callbacks=[calls_.json_logging_callback,
                                            calls_.slack_callback,
                                            get_model_checkpoint(),get_Tensorboard()], steps_per_epoch =steps_per_epoch_,
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
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # CHANGED THIS!!!!
        input_data = input_data / 255
        predictions = self.model.predict(input_data, verbose=False)
        return np.array(predictions[0])


    def return_weights(self,layer):
        return self.model.layers[layer].get_weights()



class cycle_model:
    def __init__(self):
        self.var = 0
        # x,x_vel,x_acc,time
        self.history = np.zeros((1,8))#BayesianRidge
        self.models = {"x":linear_model.LinearRegression(),"y":linear_model.LinearRegression(),"z":linear_model.LinearRegression()}
        self.poly = PolynomialFeatures(degree=2)

    def add_points(self,pose,time):
        time = float(time)/10**9
        x,y,z = pose[1]
        dx,dy,dz = pose[2][:3]
        new_vals = np.zeros((1,8))
        if self.history.shape[0] > 1:
            since_start = time-self.history[1,6]
        else:
            since_start = 0

        new_vals[:] = x,y,z,dx,dy,dz,time,since_start
        self.history = np.concatenate([self.history,new_vals],axis=0)


    def fit(self,gap):
        features = np.zeros(((self.history.shape[0]-1)*3-3*gap,self.history.shape[1]-1))

        '''print("History is " + str(self.history))
        print("Gap is " + str(gap))'''
        for i,z in zip(range(0,features.shape[0],3),range(0,self.history.shape[0]-1,1)):
            for pert in range(3):
                i_1,i_2 = int(z+gap+pert-1),z
                f_index = i+pert
                features[f_index,-1] = self.history[i_1,-1] - self.history[i_2,-1]
                features[f_index,0] = self.history[i_1,0] - self.history[i_2,0]
                features[f_index,1] = self.history[i_1,1] - self.history[i_2,1]
                features[f_index,2] = self.history[i_1,2] - self.history[i_2,2]
        '''
        print("Features are : " + str(features[:,:]))
        print("Xs are : " + str(features[:,[0,1,2]]))
        print("Max features are : " + str(np.max(features[:-gap,3:])))
        print("Min features are : " + str(np.min(features[:-gap,3:])))
        print(" X values are : " + str(features[:,0]))'''
        vars = self.poly.fit_transform(features[3:,3:].reshape((-1,4)))
        self.models["x"].fit(vars,features[3:,0].reshape((-1,1)))
        self.models["y"].fit(vars,features[3:,1].reshape((-1,1)))
        self.models["z"].fit(vars,features[3:,2].reshape((-1,1)))


    def predict(self,time):
        #time_lapse = time - self.history[-1,6]
        if self.history.shape[0] < 3:
            return self.history[-1,[3,4,5]]*time#/10**9
        gap = 0
        for i,val in enumerate(self.history[1:,-1]):
            #print("val is " + str(val) + "time is " + str(time))
            if val >= time:
                gap = i
                break
        if gap == 0 or self.history.shape[0]-gap < 4:
            #print("Gap equals zero: returning " + str(self.history[-1,[3,4,5]]*time))
            return self.history[-1,[3,4,5]]*time#/10**9


        self.fit(gap)
        vars = self.history[-1,3:-2].tolist()
        vars.append(time)
        #print("vars are : " + str(np.array(vars).reshape((-1,4))))
        vars= self.poly.fit_transform(np.array(vars).reshape((-1,4)))
        x = self.models['x'].predict(vars) + self.history[-1,0]
        y = self.models['y'].predict(vars) + self.history[-1,1]
        z = self.models['z'].predict(vars) + self.history[-1,2]
        #print("Prediction for time " + str(time) + " is " + str([x,y,z]))
        return x,y,z
