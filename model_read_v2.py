from keras.models import load_model
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, LSTM, MaxPooling2D, CuDNNLSTM, Bidirectional
from keras.optimizers import Adam
import h5py
import functions_dan as fn
import numpy as np
import blankSlate
import gc
import csv


# Load Relevant Files for the trained net and the testing sets

def MAEnp(y_true, y_pred):
    return np.mean(abs(y_pred-y_true)/abs(y_true))

def MAEnp_vector(y_true, y_pred):
    return (abs(y_pred-y_true)/abs(y_true))

def printArray(arr,w,h,a):
    for x in range(0,h):
        print a,x,
        for y in range(0,w):
            print arr[x,y],
        print()
def printLayout(arr,w,h,a):
    #print a
    #print("000,000",arr[0,0][3],arr[0,0][4])
    #print("800,000",arr[0,w-1][3],arr[0,w-1][4])
    print("400,400", arr[int(w/2),int(h/2)])#arr[int(w/2),int(h/2)][2],arr[int(w/2),int(h/2)][3],arr[int(w/2),int(h/2)][4])
    #print("000,800",arr[h-1,0][3],arr[h-1,0][4])
    #print("800,800 ", arr[w-1,h-1][3], arr[w-1,h-1][4])

#initial variables
trainedNet_file = '/home/dfinolbe/PCNN/LSTM/trained_nets/file_003_3tsteps_60Hz_6ksamp_800max0mean_v3_[4275].h5'
input_filename1_norm = '/home/dlopezmo/datasetsV3/100box/ball_in_polygon_6000v3_tbox100.h5'
output_filename1_norm = '/home/dlopezmo/datasetsV3/100box/xyrot_ball_in_polygon_6000v3.h5'
test_set1 = '/home/dlopezmo/datasetsV3/full_frame/full_ball_in_polygon_6000v3.h5'
inp2 = '/home/dlopezmo/datasetsV3/100box/ball_in_polygon_6000v3_tbox100.h5'
out2 = '/home/dlopezmo/datasetsV3/100box/xyrot_ball_in_polygon_6000v3.h5'

t_steps = 3 # Tensor time step reshaping
diff_step = 0
axis = 0
n_train = 2800
n_test = 2000
xymax = 800
pixelmax = 255
norm_axis = (0,1,2,3,4)
size = 100
offset = 0
#origin_offset = 3
#minPredict = 90
#full_len = 960



## Load Normalization dataset
x_data1 = h5py.File(input_filename1_norm, 'r')
x_data1 = x_data1['matrices']
y_data1 = h5py.File(output_filename1_norm, 'r')
y_data1 = y_data1['matrices']
y_data1 = y_data1[:,:2]

x_data2 = h5py.File(inp2, 'r')
x_data2 = x_data2['matrices']
y_data2 = h5py.File(out2, 'r')
y_data2 = y_data2['matrices']
y_data2 = y_data2[:,:2]
#for ys in y_data2:
#    print(ys)

nsteps_tmarch = 30
offset_tsteps = 20
xymax = 800
xmax = xymax
xmean = 0
subsamp_ratio = 2
x_data2 = x_data2[offset_tsteps:]
y_data2 = y_data2[offset_tsteps:]

img = blankSlate.changeFrame(size, test_set1)
model = load_model(trainedNet_file)

#Reshape data

x_data1 = x_data1[np.arange(0,x_data1.shape[0],subsamp_ratio),:,:,:]
x_data2 = x_data2[np.arange(0,x_data2.shape[0],subsamp_ratio),:,:,:]
y_data2 = y_data2[np.arange(0,x_data2.shape[0],subsamp_ratio)]
x_set1 = fn.sequence_input3(x_data1,t_steps,axis,diff_step)
x_set2 = fn.sequence_input3(x_data2,t_steps,axis,diff_step)
#print('Post Seq')
#print(x_set1.shape)

#separate training and test sets
x_train = x_set1[:n_train,:,:,:,:]  #
x_test2 = x_set2[2:200]
#Normalize data
xymax = 800
xmax = xymax
xmean = 0
xnorm_type = 'mean'
x_train,xdata_mean,xdata_range = fn.data_normalize_v2(x_train,norm_type= xnorm_type,data_mean= xmean,data_range=xmax,norm_axis=norm_axis)

x_test2 = fn.data_normalize_v2(x_test2,norm_type= xnorm_type ,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)[0]


#Predict
predicted = (model.predict_on_batch(x_test2)+offset)*800
print(predicted)
