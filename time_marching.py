from keras.models import load_model
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, LSTM, MaxPooling2D, CuDNNLSTM, Bidirectional
from keras.optimizers import Adam
import h5py
import functions as fn
import numpy as np
import blankSlate
import gc

# Load Relevant Files for the trained net and the testing sets

print_all = False
print_shapes = False
print_data = False
print_real = False
check_error = False

def printLayout(arr,w,h,a):
    #print a
    #print("000,000",arr[0,0][3],arr[0,0][4])
    #print("800,000",arr[0,w-1][3],arr[0,w-1][4])
    print(a,"400,400", arr[int(w/2),int(h/2)])#arr[int(w/2),int(h/2)][2],arr[int(w/2),int(h/2)][3],arr[int(w/2),int(h/2)][4])
    #print("000,800",arr[h-1,0][3],arr[h-1,0][4])
    #print("800,800 ", arr[w-1,h-1][3], arr[w-1,h-1][4])

#initial variables

trainedNet_file = 'file_003_3tsteps_120Hz_[3243].h5'
input_filename1 = '/home/dlopezmo/datasetsV2/100box/ball_in_polygon_3ballsv2_tbox100.h5'
output_filename1 = '/home/dlopezmo/datasetsV2/100box/xyrot_ball_in_polygon_3ballsv2.h5'
t_steps = 3 # Tensor time step reshaping
diff_step = 0
axis = 0
n_train = 750
n_test = 800
xymax = 800
pixelmax = 255
norm_axis = (0,1,2,3,4)
size = 100
offset = 0
origin_offset = 3
minPredict = 90
full_len = 960
x_data1 = h5py.File(input_filename1, 'r')
x_data1 = x_data1['matrices']
y_data1 = h5py.File(output_filename1, 'r')
y_data1 = y_data1['matrices']
y_data1 = y_data1[:,:2]


for y in y_data1[n_test-15:n_test+5]:
    print(y)

#call new frame class and model
img = blankSlate.changeFrame(size, 'datasetsV2/full_frame/full_ball_in_polygon_3ballsv2.h5')
model = load_model(trainedNet_file)

#Reshape data
subsamp_ratio = 1
x_data1 = x_data1[np.arange(0,x_data1.shape[0],subsamp_ratio),:,:,:]
print("Xdata1",x_data1.shape)
x_set1= fn.sequence_input3(x_data1,t_steps,axis,diff_step)


#separate training and test sets
x_train=x_set1[:n_train,:,:,:,:]  #s_train=x_set1[:n_train,:,nred:100-nred,nred:100-nred,:] #print(np.all(np.equal(x_train,s_train))) #nred = 0 #when nred = 0 x_train == s_train
x_test=x_set1[n_train:n_test,:,:,:,:]
'''
printLayout(x_test[len(x_test)-3][0], 100, 100, "Set[00]")
printLayout(x_test[len(x_test)-3][1], 100, 100, "Set[01]")
printLayout(x_test[len(x_test)-3][2], 100, 100, "Set[02]")
printLayout(x_test[len(x_test)-2][0], 100, 100, "Set[10]")
printLayout(x_test[len(x_test)-2][1], 100, 100, "Set[11]")
printLayout(x_test[len(x_test)-2][2], 100, 100, "Set[12]")
printLayout(x_test[len(x_test)-1][0], 100, 100, "Set[20]")
printLayout(x_test[len(x_test)-1][1], 100, 100, "Set[21]")
printLayout(x_test[len(x_test)-1][2], 100, 100, "Set[22]")
'''
#Normalize data
xnorm_type = 'mean'
x_train,xdata_mean,xdata_range = fn.data_normalize_v2(x_train,norm_type= xnorm_type,data_mean='non',data_range='non',norm_axis=norm_axis)
norm_test = fn.data_normalize_v2(x_test,norm_type= xnorm_type ,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)[0]
#print("1x_test",x_test.shape)

gc.collect()

#Predict
predicted = (model.predict_on_batch(norm_test[len(x_test)-4:,:,:,:,:])+offset)*800
print(predicted)
x = int(round(predicted[len(predicted)-4][0]))
y = int(predicted[len(predicted)-1][1])
print x,',', y

frame = np.array([img.output(n_test, x, y)])
print(x_test[1:].shape,frame.shape)
mod_data = np.concatenate((x_data1,frame))
x_set1= fn.sequence_input3(mod_data,t_steps,axis,diff_step)
x_test=x_set1[n_train+1:n_test+1,:,:,:,:]
x_test = fn.data_normalize_v2(x_test,norm_type= xnorm_type ,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)[0]
#print("X_test2",x_test.shape)


gc.collect()

#Predict
predicted = (model.predict_on_batch(x_test[len(x_test)-4:,:,:,:,:])+offset)*800
print(predicted)
x = int(round(predicted[len(predicted)-3][0]))
y = int(predicted[len(predicted)-1][1])
print x,',', y


offsets = -4
mod_data = x_data1[n_test-6-offsets:n_test-offsets]

for index in range(2,100):
    frame = np.array([img.output(n_test, x, y)])
    #printLayout(frame[0], 100, 100, "Set[00]")
    mod_data = np.concatenate((mod_data[1:],frame))
    '''
    for data in mod_data:
       printLayout(data,100,100,"set0")
    '''
    #print(mod_data.shape)
    x_set1= fn.sequence_inputs((mod_data))
    x_test=x_set1
    #print("X_test2", x_test.shape)

    printLayout(x_test[0][0], 100, 100, "Set[00]")
    printLayout(x_test[0][1], 100, 100, "Set[01]")
    printLayout(x_test[0][2], 100, 100, "Set[02]")
    print()
    printLayout(x_test[1][0], 100, 100, "Set[10]")
    printLayout(x_test[1][1], 100, 100, "Set[11]")
    printLayout(x_test[1][2], 100, 100, "Set[12]")
    print()
    printLayout(x_test[2][0], 100, 100, "Set[20]")
    printLayout(x_test[2][1], 100, 100, "Set[21]")
    printLayout(x_test[2][2], 100, 100, "Set[22]")
    print()
    printLayout(x_test[3][0], 100, 100, "Set[30]")
    printLayout(x_test[3][1], 100, 100, "Set[31]")
    printLayout(x_test[3][2], 100, 100, "Set[32]")

    x_test = fn.data_normalize_v2(x_test,norm_type= xnorm_type ,data_mean=xdata_mean,data_range=xdata_range,norm_axis=norm_axis)[0]

    gc.collect()

    #Predict
    predicted = (model.predict_on_batch(x_test[len(x_test)-4:,:,:,:,:])+offset)*800
    print(predicted)
    x = int(round(predicted[len(predicted)-3][0]))
    y = int(predicted[len(predicted)-1][1])
    print x,',', y
