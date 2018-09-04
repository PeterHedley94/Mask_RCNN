import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')

from utils.folder_manipulation import *
#from src.PREPROCESSING.preprocessing import *
#from keras.utils import to_categorical
from PIL import Image
import json





# Class to yield batches of image data indefinitely for as many epochs as required
class DataGenerator(object):
    def __init__(self,dir, batch_size = 16, shuffle = True, sequence_length = 1, time_distributed = False, debug_mode = False,images = False,
                                                                    image_height = 0,image_width=0,no_image_channels=0,camera_model_location = os.path.join("utils",'camera_model.json')):
        print("New instance of 'datagenerator' initialised on {}".format(dir))
        # Attributes initialised directly from arguments

        self.image_height = image_height
        self.image_width = image_width
        self.no_image_channels = no_image_channels

        self.n_outputs = 3 #dx,dy,dz
        self.data_length = 12 #3x4 camera matrix
        self.time_distributed_ = time_distributed
        self.debug_mode_ = debug_mode
        self.use_images = images
        if self.use_images:
            print("Images enabled!!")

        self.shuffle = shuffle
        self.sequence_length_ = sequence_length

        self.epoch_number = 1
        with open(camera_model_location) as f:
            self.camera_model = json.load(f)

        #Central point matrix [c1,c2]
        self.c_mat = np.zeros((3,1))
        self.c_mat[1,:],self.c_mat[0,:] = self.camera_model["principal_point"]
        #Focal Matrix [[1/f1,0],[0,1/f2]]
        self.foc_mat = np.zeros((2,2))
        self.foc_mat[0,0],self.foc_mat[1,1] = 1/self.camera_model["focal_length"][0],1/self.camera_model["focal_length"][1]
        #Focal Matrix Inv [[1/f1,0],[0,1/f2]]
        self.foc_mat_inv = np.zeros((2,2))
        self.foc_mat_inv[0,0],self.foc_mat_inv[1,1] = self.camera_model["focal_length"][0],self.camera_model["focal_length"][1]
        #Camera Transformed
        self.T_SC = np.array(self.camera_model["T_SC"]).reshape((4,4))

        # Produce lists of all the image sequence files in the folders (trainx)
        # and corresponding one-hot array labels (trainy)
        self.file_numbers,self.T_WS_C,self.T_WS_r,self.sb,self.images = self.get_files(dir)

        # Batch size is defined here as the size of the first dimension of the output
        # If not time distributed this is the number of images in the batch
        # If time distributed there will be batch_size*sequence_length images per batch
        self.batch_size = batch_size
        self.effective_batch_size = batch_size
        self.index_buffer = self.sequence_length_+1
        #CHENGED TO if ... instead of if not ...?
        # Set the number of batches per epoch to iterate over in generation
        print("Number of batches per epoch is " + str((len(self.file_numbers)-self.index_buffer) / self.batch_size))
        self.batches_per_epoch =  int(np.floor((len(self.file_numbers)-self.index_buffer) / self.batch_size))


    def generate(self):
        # For as many epochs as required:
        while 1:
            #dont want sequence near end
            indexes = self.__get_exploration_order(self.file_numbers[:-self.index_buffer])
            # Generate batches
            for i in (range(self.batches_per_epoch)):
                batch_indexes = indexes[i*self.effective_batch_size:(i+1)*self.effective_batch_size]
                yield(self.__data_generation(batch_indexes))
            self.epoch_number+=1

    def gd(self,filename):
        return np.loadtxt(filename)

    def to_number(self,item):
        item = item.split("/")[-1]
        number = int(item.split("_")[0])
        return number

    def world_to_camera(self,array):
        #array is (3,N) containing x,y,z
        N = array.shape[1]
        world_point = np.ones((4,N))
        world_point[:3,:] = array
        camera_point = np.matmul(self.T_SW,world_point)
        camera_point[:3,:] = camera_point[:3,:]/camera_point[2,:]
        camera_point[:2,:] = np.matmul(self.foc_mat_inv,camera_point[:2,:])
        camera_point[:3,:] = camera_point[:3,:]+self.c_mat
        return camera_point[:3,:]

    def camera_to_world(self,array):
        #array is (3,N)
        N = array.shape[1]
        camera_point = np.ones((4,N))
        camera_point[:3,:] = array
        camera_point[:3,:] = camera_point[:3,:]-self.c_mat
        camera_point[:2,:] = np.matmul(self.foc_mat,camera_point[:2,:])
        camera_point[:2,:] = camera_point[:2,:]*camera_point[2,:]
        world_point = np.matmul(self.T_WS,camera_point)[:3,:]
        return world_point


    #Get lists of the image folders and labels
    def get_files(self,dir):
        data_folder = os.path.join(dir,"pose")
        o_T_WS_C = get_file_names(data_folder,"T_WS_C.txt",self.to_number)
        o_T_WS_r = get_file_names(data_folder,"T_WS_r.txt",self.to_number)
        o_sb = get_file_names(data_folder,"sb.txt",self.to_number)
        file_numbers = np.arange(len(o_T_WS_C))
        images = get_image_names(os.path.join(dir,"cam0","data"))
        return file_numbers,o_T_WS_C,o_T_WS_r,o_sb,images

    def __get_exploration_order(self, list_IDs):
        indexes = np.arange(len(list_IDs)-1)+1 #dont use first one
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes

    def get_T_WC(self,T_WS_C,T_WS_r):
        T_WS = np.zeros((4,4),dtype = np.float64)
        T_WS[:3,:4] = np.concatenate((T_WS_C,T_WS_r[:,None]),axis = 1)
        T_WS[3,3] = 1
        T_WC = np.matmul(T_WS,self.T_SC)
        return T_WC

    def world_to_camera(self,array,T_CW):
        #array is (3,N) containing x,y,z
        N = array.shape[1]
        world_point = np.ones((4,N))
        world_point[:3,:] = array
        camera_point = np.matmul(T_CW,world_point)
        camera_point[:2,:] = camera_point[:2,:]/camera_point[2,:]
        camera_point[:2,:] = np.matmul(self.foc_mat_inv,camera_point[:2,:])
        camera_point[:3,:] = camera_point[:3,:]+self.c_mat
        return camera_point[:3,:]

    def reverse_transform(self,array):
        reversed = np.zeros((4,4),dtype = np.float64)
        reversed[:3,:4] = np.concatenate((array[:3,:3].transpose(),
                                                np.matmul(-array[:3,:3].transpose(),array[:3,3:])),axis = 1)
        reversed[3,3] = 1
        return reversed

    # Get the data and labels for a batch of folders
    def __data_generation(self,batch_indexes):

        X = None
        X = np.zeros(shape=(self.batch_size, self.sequence_length_,self.data_length))
        Y = np.zeros(shape=(self.batch_size, self.n_outputs))

        if self.debug_mode_:
            X_initial_points = np.zeros(shape=(self.batch_size, self.n_outputs))

        if self.use_images:
            X_images = np.zeros(shape=(self.batch_size, self.sequence_length_,self.image_height,self.image_width,self.no_image_channels))
        # Loop over the input: make call to dstack then append result with label:
        k = 0
        for seq_no in range(self.effective_batch_size):
            #print("NEW SEQUENCE")
            index = batch_indexes[seq_no]
            i_T_W_CL1 = self.get_T_WC(self.gd(self.T_WS_C[index]),self.gd(self.T_WS_r[index]))[:3,:]
            if self.debug_mode_:
                X_initial_points[seq_no,:,None] = i_T_W_CL1[:3,3:].reshape((-1,1))

            if self.use_images:
                print("Getting image " + str(self.images[index]) + " \n ")
                X_images[seq_no,0,:,:,:] = get_resized_image(self.images[index],height=self.image_height,width=self.image_width)

            for no_in_seq in range(1,self.sequence_length_,1):
                T_W_C_next = self.get_T_WC(self.gd(self.T_WS_C[index+no_in_seq]),self.gd(self.T_WS_r[index+no_in_seq]))[:3,:]
                #only get difference for translation!!
                T_W_C_next[:3,3:] = T_W_C_next[:3,3:]-i_T_W_CL1[:3,3:]
                #print("T_W_C difference is " + str(T_W_C_difference))
                T_W_C_difference = T_W_C_next.reshape((1,-1)).tolist()[0]
                X[seq_no,no_in_seq,:] = T_W_C_difference

                #Images
                if self.use_images:
                    X_images[seq_no,no_in_seq,:,:,:] = get_resized_image(self.images[index+no_in_seq],self.image_height,self.image_width)

            #Add first image in sequence
            #Place origin at start of sequence
            i_T_W_CL1[:3,3:] = np.array([0,0,0]).reshape((3,1))
            X[seq_no,0,:] = i_T_W_CL1.reshape((1,-1))

            #Add Y data
            y_index = index+self.sequence_length_
            y_T_W_r = self.gd(self.T_WS_r[y_index]).reshape((3,-1))-i_T_W_CL1[:3,3:]
            #print(i_T_W_CL1[:3,3:])
            #i_T_CL1_w = self.reverse_transform(i_T_W_CL1)
            #y_camera_point = self.world_to_camera(y_T_W_r,i_T_CL1_w)
            #y_camera_point[2,0] = y_T_W_r[2,0]
            #print(y_camera_point)
            Y[seq_no,:,None] = y_T_W_r.reshape((-1,1))
        if self.use_images:
            X_images = (X_images - 127.5) / 127.5
            if self.debug_mode_:
                return X_initial_points,X_images,X,Y
            else:
                return X_images,X,Y
        else:
            if self.debug_mode_:
                return X_initial_points,X,Y
            return X,Y
