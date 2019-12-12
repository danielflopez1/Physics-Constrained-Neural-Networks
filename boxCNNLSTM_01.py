""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
import numpy as np
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, LSTM, MaxPooling2D, CuDNNLSTM
#from keras.utils import np_utils
#from keras.constraints import maxnorm
#from keras.optimizers import SGD
from keras.optimizers import Adam
#from keras.layers.normalization import BatchNormalization
import h5py 
import functions as fn 
#from sklearn.preprocessing import MinMaxScaler

print('Loading data..')
# fix random seed for reproducibility
#np.random.seed(7)
# Load Data 

input_filename = 'datasets/collision/ballMatrices.h5'
output_filename = 'datasets/collision/ballProperties.h5'
input_filename2 = 'datasets/collision/ballMatrices2.h5'
output_filename2 = 'datasets/collision/ballProperties2.h5'
input_filename3 = 'datasets/projectile/ballMatrices.h5'
output_filename3 = 'datasets/projectile/ballProperties.h5'

x_data = h5py.File(input_filename, 'r')
x_data = x_data['matrices']
y_data = h5py.File(output_filename, 'r')
y_data = y_data['matrices']
y_data = y_data[:,:2]

x_data2 = h5py.File(input_filename2, 'r')
x_data2 = x_data2['matrices']
y_data2 = h5py.File(output_filename2, 'r')
y_data2 = y_data2['matrices']
y_data2 = y_data2[:,:2]

x_data3 = h5py.File(input_filename3, 'r')
x_data3 = x_data3['matrices']
y_data3 = h5py.File(output_filename3, 'r')
y_data3 = y_data3['matrices']
y_data3 = y_data3[:,:2]


print('Preprocessing Input..')
t_steps = 1  
diff_step = 1
axis = 0
x_set1 = np.reshape(x_data,(x_data.shape[0],1,x_data.shape[1],x_data.shape[2],x_data.shape[3]))
x_set2 = np.reshape(x_data2,(x_data2.shape[0],1,x_data2.shape[1],x_data2.shape[2],x_data2.shape[3]))
x_set3 = np.reshape(x_data3,(x_data3.shape[0],1,x_data3.shape[1],x_data3.shape[2],x_data3.shape[3]))
x_set3 = x_set3[0:x_set1.shape[0]-1,:]

# For single future step prediction

y_set1 = y_data[t_steps:y_data.shape[0],:]
y_set2 = y_data2[t_steps:y_data2.shape[0],:]
y_set3 = y_data3[t_steps:y_data3.shape[0],:]

x_train = np.append(x_set1[0:x_set1.shape[0]-1,:],x_set2[0:x_set2.shape[0]-1,:],axis=axis)
y_train = np.append(y_set1,y_set2,axis=axis)

#np.savez_compressed('datatemp', x_train=x_train, y_train=y_train)
#quit()
#datanpz = np.load('datatemp.npz')
#x_train = datanpz['x_train']
#y_train = datanpz['y_train']

# Normalization 

xymax = 800
pixelmax = 255

#x_train,xdata_mean,xdata_range = fn.data_normalize(x_train,norm_type='mean',data_mean=None,data_range=None)
y_train,ydata_mean,ydata_range = fn.data_normalize(y_train,norm_type='max',data_mean=0,data_range=xymax)
#x_set3,xdata_mean,xdata_range = fn.data_normalize(x_set3,norm_type='mean',data_mean=xdata_mean,data_range=xdata_range) 
y_set3,ydata_mean,ydata_range = fn.data_normalize(y_set3,norm_type='max',data_mean=ydata_mean,data_range=xymax)

# Norm independent per component (pixels/255 and x-y/800)
x_train[:,:,:,0:3] = x_train[:,:,:,0:3]/pixelmax
x_train[:,:,:,3:5] = x_train[:,:,:,3:5]/xymax
x_set3[:,:,:,0:3] = x_set3[:,:,:,0:3]/pixelmax
x_set3[:,:,:,3:5] = x_set3[:,:,:,3:5]/xymax

# Norm using sklearn 

#scaler = MinMaxScaler(feature_range=(0, 1))
#x_train = scaler.fit_transform(x_train)
#y_train = scaler.fit_transform(y_train)
#x_set3 = scaler.fit_transform(x_set3)
#y_set3 = scaler.fit_transform(y_set3)


#y_train = np.append(y_data[t_steps+diff_step:y_data.shape[0]-diff_step,:],y_data2[t_steps+diff_step:y_data.shape[0]-diff_step,:],axis=axis)
#y_train_one = 
#x_train = x_train/np.amax(x_train)
#y_train = y_train/np.amax(y_train)
print(x_data2.shape)
print(x_set2.shape)
print(x_train.shape)
print(y_data2.shape)
print(y_set2.shape)
print(y_train.shape)
#print(y_data[900:910,:])

#quit()

##### CNN-LSTM Parameter Control Module #####

n_neurons = 3
n_hidden_units = 100
n_convpool_layers = 1
n_convlayers = 3
n_reglayers = 1
max_poolsize = (2,2)
train_set_ratio = 0.7
valid_set_ratio = 0.15
drop_rate = 0.0
n_filters = 100
kernel_size = (2,2)
input_strides = 1
kernel_init = 'uniform'
cost_function = 'mean_squared_error'
batch_size = 1
n_epochs = 50
n_output = 2
n_post_test = 5000
early_stop_delta = 0.0005 # 0.0025 change or above is considered improvement
early_stop_patience = 10 # keep optimizing for 10 iterations under "no improvement"
out_filepath = 'file_003.h5'

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels(RxGxBxXxY)) and returns the X,Y
# coordinates of moving object

model = Sequential()

# define CNN model architecture


model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
                  kernel_initializer= kernel_init,
                  activation='relu'),input_shape=(1,100,100,5)))
model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

#model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
#                  kernel_initializer= kernel_init,
#                  activation='relu')))
#model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

model.add(TimeDistributed(Flatten()))

model.add(TimeDistributed(Dense(100, activation='relu',kernel_initializer= kernel_init)))
model.add(TimeDistributed(Dense(100, activation='relu',kernel_initializer= kernel_init)))
#model.add(TimeDistributed(Dense(100, activation='relu')))
#model.add(TimeDistributed(Dense(150, activation='relu')))
#model.add(TimeDistributed(Dense(10, activation='linear',kernel_initializer= kernel_init)))

# Automatic Setup

#for ii in range (n_convpool_layers):
# for i in range(n_convlayers):
#  model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
#                  kernel_initializer= kernel_init,
#                  activation='relu'),input_shape=(3,100,100,5)))
# model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

#model.add(TimeDistributed(Flatten()))

# define LSTM model
model.add(CuDNNLSTM(n_neurons, return_sequences=True,kernel_initializer= kernel_init))
model.add(CuDNNLSTM(n_neurons,kernel_initializer= kernel_init))
#for i in range(n_reglayers):
# model.add(TimeDistributed(Dense(n_hidden_units,activation='relu',kernel_initializer= kernel_init)))

model.add(Dense(n_hidden_units,activation='relu',kernel_initializer= kernel_init))
model.add(Dense(n_hidden_units,activation='relu',kernel_initializer= kernel_init))

model.add(Dense(n_output, activation='linear',kernel_initializer= kernel_init))

model.summary()

def MAE(y_true, y_pred):
    return K.mean(K.abs(y_pred-y_true)/K.abs(y_true))

model.compile(loss=cost_function, optimizer='adam', metrics=[MAE])

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=early_stop_delta, patience=early_stop_patience, verbose=1, mode='auto')

# Train the network

history = model.fit(x_train, y_train, epochs=n_epochs,
        batch_size=batch_size, verbose=1,
        callbacks=[earlyStopping],
        validation_split = 0.2)
#        validation_data =[inp_valid,out_valid])

model.save(out_filepath)

score = model.evaluate(x_set3, y_set3, verbose=0)
print 'Test loss:', score[0]
print 'Test error:', score[1]

## Post-Processing
print('Actual Values')
print(y_train[110:112,:])
print('Predicted Values')
print((model.predict_on_batch(x_train[110:112,:,:,:,:])))

## Post-Processing
print('Actual Values')
print(y_set3[10:12,:])
print('Predicted Values')
print((model.predict_on_batch(x_set3[10:12,:,:,:,:])))


# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions

#for j in range(16):
#    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
#    new = new_pos[::, -1, ::, ::, ::]
#    track = np.concatenate((track, new), axis=0)



