import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
#parentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir)
sys.path.insert(1,'/usr/local/lib/python3.5/dist-packages')
import numpy as np,random
import math,cv2
from MovementModels.cyclist_rnn_model import *
from MovementModels.data_generator import *
from utils.pose_visualiser import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# = CNN_LSTM()#cached_model="/home/peter/catkin_ws/src/mask_rcnn/src/MODEL_OUTPUTS/models/vgg_netfirst_try.hdf5")

train_directory_ = "/home/peter/Documents/okvis_drl/build/tate3_dataset"
validation_directory_ = "/home/peter/Documents/okvis_drl/build/tate3_dataset"
model_description,epochs =  "first_try",10
#cnn.train(train_directory_, validation_directory_,model_description,epochs)

COUNT = 0
no_files = 1000

def create_test_data_structure(directory):
    dirs = [directory,os.path.join(directory,"pose"),
    os.path.join(directory,"cam0"),os.path.join(directory,"cam0","data")]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def create_circle_yaw_test_data(directory,global_):
    global COUNT
    radius = 52
    angle = 0
    count = 0
    if global_:
        count = COUNT
    for i in range(no_files):
        if i > 0:
            angle_ = i*(360+90)/1000
        else:
            angle_ = 0
        rad = math.radians(angle_)
        x = radius*math.cos(rad)
        y = radius*math.sin(rad)
        C = np.array([[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]]).reshape((3,3))
        r = np.array([x,y,0]).reshape((3,-1))
        r_f = os.path.join(directory,"pose",str(count) + "_T_WS_r.txt")
        c_f = os.path.join(directory,"pose",str(count) + "_T_WS_C.txt")
        s_f = os.path.join(directory,"pose",str(count) + "_sb.txt")
        np.savetxt(r_f,r, delimiter=' ')
        np.savetxt(c_f,C, delimiter=' ')
        np.savetxt(s_f,r, delimiter=' ')
        count+= 1
    if global_:
        COUNT = count

def create_circle_pitch_test_data(directory,global_):
    global COUNT
    radius = 52
    angle = 0
    count = 0
    if global_:
        count = COUNT
    for i in range(no_files):
        if i > 0:
            angle_ = i*(360+90)/1000
        else:
            angle_ = 0
        rad = angle_*3.14/180
        x = radius*math.cos(rad)
        y = radius*math.sin(rad)
        C = np.array([[math.cos(rad),0,math.sin(rad)],[0,1,0],[-math.sin(rad),0,math.cos(rad)]]).reshape((3,3))
        r = np.array([0,x,y]).reshape((3,-1))
        r_f = os.path.join(directory,"pose",str(count) + "_T_WS_r.txt")
        c_f = os.path.join(directory,"pose",str(count) + "_T_WS_C.txt")
        s_f = os.path.join(directory,"pose",str(count) + "_sb.txt")
        np.savetxt(r_f,r, delimiter=' ')
        np.savetxt(c_f,C, delimiter=' ')
        np.savetxt(s_f,r, delimiter=' ')
        count+= 1
    if global_:
        COUNT = count



def create_circle_roll_test_data(directory,global_):
    global COUNT
    radius = 52
    angle = 0
    count = 0
    if global_:
        count = COUNT
    for i in range(no_files):
        if i > 0:
            angle_ = i*(360+90)/1000
        else:
            angle_ = 0
        rad = angle_*3.14/180
        x = radius*math.sin(rad)
        z = radius*math.cos(rad)
        C = np.array([[1,0,0],[0,math.cos(rad),-math.sin(rad)],[0,math.sin(rad),math.cos(rad)]]).reshape((3,3))
        r = np.array([x,0,z]).reshape((3,-1))
        r_f = os.path.join(directory,"pose",str(count) + "_T_WS_r.txt")
        c_f = os.path.join(directory,"pose",str(count) + "_T_WS_C.txt")
        s_f = os.path.join(directory,"pose",str(count) + "_sb.txt")
        np.savetxt(r_f,r, delimiter=' ')
        np.savetxt(c_f,C, delimiter=' ')
        np.savetxt(s_f,r, delimiter=' ')
        count+= 1
    if global_:
        COUNT = count


def create_test_data(directory):
    count = 0
    for i in range(no_files):
        C = np.array([[1,0,0],[0,1,0],[0,0,1]]).reshape((3,3))
        r = np.array([count,count,count]).reshape((3,-1))
        r_f = os.path.join(directory,"pose",str(count) + "_T_WS_r.txt")
        c_f = os.path.join(directory,"pose",str(count) + "_T_WS_C.txt")
        s_f = os.path.join(directory,"pose",str(count) + "_sb.txt")
        np.savetxt(r_f,r, delimiter=' ')
        np.savetxt(c_f,C, delimiter=' ')
        np.savetxt(s_f,r, delimiter=' ')
        count+=1


directory = "/home/peter/Tests"
create_test_data_structure(directory)
directory = "/home/peter/Tests"#"/home/peter/Documents/okvis_drl/build/tate3_dataset"
train_directory_,validation_directory_ = directory,directory

images = False
image_height = 480#180
image_width= 640#240
no_image_channels=3
camera_model_location = os.path.join("utils",'test_camera_model.json')
sequence_length = 50

params_train = {'dir': train_directory_,
              'batch_size': 1,
              'shuffle': True,'debug_mode':True,
                'sequence_length': sequence_length ,'time_distributed' : True,
                'images':images,'image_height':image_height,'image_width':image_width,
                'no_image_channels':no_image_channels,'camera_model_location':camera_model_location}

#create_test_data,

for test in [create_circle_roll_test_data]:#create_circle_pitch_test_data,create_circle_roll_test_data,create_circle_yaw_test_data]:#,create_circle_pitch_test_data]:
    test(directory,True)

'''
class future_points_visualiser:
    def __init__(self,camera_model_location):
        with open(camera_model_location) as f:
            self.camera_model = json.load(f)
        self.height = self.camera_model["image_dimension"][1]
        self.width = self.camera_model["image_dimension"][0]
        #Central point matrix [c1,c2]
        self.c_mat = np.zeros((3,1))
        self.c_mat[0,:],self.c_mat[1,:] = self.camera_model["principal_point"]
        #Focal Matrix [[1/f1,0],[0,1/f2]]
        self.foc_mat = np.zeros((2,2))
        self.foc_mat[0,0],self.foc_mat[1,1] = 1/self.camera_model["focal_length"][0],1/self.camera_model["focal_length"][1]
        #Focal Matrix Inv [[1/f1,0],[0,1/f2]]
        self.foc_mat_inv = np.zeros((2,2))
        self.foc_mat_inv[0,0],self.foc_mat_inv[1,1] = self.camera_model["focal_length"][0],self.camera_model["focal_length"][1]
        #Camera Transformed
        self.T_SC = np.array(self.camera_model["T_SC"]).reshape((4,4))


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

    def plot_image(self,img,coords,initial_transform):
        T_CW = self.reverse_transform(np.matmul(self.T_SC,initial_transform))
        #array is (3,N) containing x,y,z
        coords = self.world_to_camera(coords,T_CW)
        for c in range(coords.shape[1]):
            coord = (int(coords[0,c]),int(coords[1,c]))
            #print("Coords are " + str((coords[1,c],coords[0,c])))
            #v2.circle(img, center, radius, color)
            #cv2.circle(img,center=(10,400),radius=5, color=(255,0,0), thickness=1, lineType=8, shift=0)
            cv2.circle(img,center=coord,radius=5, color=(255,0,0), thickness=1, lineType=8, shift=0)
        return img

cnn = CNN_LSTM(lr = 0.0001,cached_model="/home/peter/catkin_ws/src/mask_rcnn/src/MODEL_OUTPUTS/models/vgg_netfirst_try.hdf5")
#cnn.train(train_directory_, validation_directory_,model_description,epochs=200)

train_generator = DataGenerator(**params_train)
train_gen = train_generator.generate()






#print("X is " + str(X))
#print("X shape is " + str(X.shape))


while 1:
    fpv = future_points_visualiser(camera_model_location)
    pv = pose_visualiser(400,400)
    if images:
        X_initial,X_images,X,Y  = train_gen.__next__()
    else:
        X_initial,X,Y = train_gen.__next__()
    for seq_no in range(X.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        #ax = fig.add_subplot(111)
        #plt.gca().set_aspect('equal', adjustable='box')
        xpoints = []
        ypoints = []
        zpoints = []
        predicted_xpoints = []
        predicted_ypoints = []
        predicted_zpoints = []

        if images:
            img = np.array(X_images[seq_no,0,:,:,:],dtype = np.uint8)
        else:
            img = np.zeros((480,640,3)) + 245

        #print("img.shape is " + str(img.shape) )
        initial_point = X_initial[seq_no,:]
        initial_transform = np.zeros((4,4))
        initial_transform[:3,:] = X[seq_no,0,:,None].reshape((3,4))
        initial_transform[:3,3:] = initial_point[:,None]
        initial_transform[3,3] = 1
        for point in range(1,sequence_length):
            point = X[seq_no,point,:,None].reshape((3,4))[:3,3:]
            point = point + initial_transform[:3,3:]
            xpoints.append(point[0,0])
            ypoints.append(point[1,0])
            zpoints.append(point[2,0])
            pv.add_points(point[[2,1],:,None])
            img = fpv.plot_image(img,point,initial_transform)

        c = pv.count
        pv.count = 1
        coord = pv.image_coords(point[[2,1],:,None].reshape((1,-1)))
        pv.count = c
        coord = (int(coord[0,0]),int(coord[0,1]))
        img2 = pv.plot(point.reshape((3,1))[:2,0,None])
        #TODO deal with this - where is initial point to predict from
        for point_i in range(0,sequence_length-4):
            initial_point = X[seq_no,point_i,:,None].reshape((3,4))[:3,3:].copy()
            points = X[seq_no,point_i:point_i+4,:,None].reshape((4,3,4)).copy()
            points[:,:3,3:] = points[:,:3,3:] - initial_point.reshape((3,1))
            point_old = cnn.predict(points.reshape((1,4,12)))
            point = point_old + initial_transform[:3,3:].reshape((3,)) + initial_point.reshape((3,))
            predicted_xpoints.append(point[0])
            predicted_ypoints.append(point[1])
            predicted_zpoints.append(point[2])
            #print("X seq point is  " + str(X[seq_no,point_i+4,:,None]))
            #print("X seq point reshaped is  " + str(X[seq_no,point_i+4,:,None].reshape((3,4))))
            X[seq_no,point_i+4,:,None][[3,7,11]] = np.array(point_old).reshape((3,1))

        print("Predicted points are : " + str(np.array([predicted_xpoints,predicted_ypoints,predicted_zpoints])))
        ax.scatter(xpoints,ypoints,zpoints, zdir='z', s=2, c='k')
        ax.scatter(predicted_xpoints,predicted_ypoints,predicted_zpoints, zdir='z', s=2, c='g')
        ax.scatter([xpoints[-1]],[ypoints[-1]], zs=[zpoints[-1]], zdir='z', s=20, c='b')
        cv2.circle(img2,center=coord,radius=5, color=(255,0,0), thickness=1, lineType=8, shift=0)
        cv2.imshow("i",img)
        cv2.imshow("i2",img2)
        plt.show()
        cv2.waitKey(0)
'''
'''
for lr in [0.0001]:#01]:
    cnn = CNN_LSTM(lr = lr)
    cnn.train(train_directory_, validation_directory_,model_description,epochs=15)
    #for i in range(X.shape[0]):


#cnn = CNN_LSTM(cached_model="/home/peter/catkin_ws/src/mask_rcnn/src/MODEL_OUTPUTS/models/vgg_netfirst_try.hdf5")
#print("X img shape is " + str(X_img.shape))
print("X shape is " + str(X.shape))
print("Y shape is " + str(Y.shape))

y_prediction = cnn.predict(X)
#y_prediction = cnn.predict(X)
print("Y prediction is " + str(y_prediction))

for i in range(X.shape[0]):
    y_prediction = cnn.predict(X[None,i,:,:])
    #y_prediction = cnn.predict(X)
    print("Y prediction is " + str(y_prediction))
print("X is " + str(X))
print("Y is " + str(Y))

'''
