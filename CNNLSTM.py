""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
import numpy as np
import keras
import keras.backend as K
from keras.self.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, LSTM, MaxPooling2D, CuDNNLSTM, Bidirectional
#from keras.utils import np_utils
#from keras.constraints import maxnorm
#from keras.optimizers import SGD
from keras.optimizers import Adam
#from keras.layers.normalization import BatchNormalization
#from sklearn.preprocessing import MinMaxScaler
import h5py
import functions as fn
import csv

class CNNLSTM:
    def __init__(self):
        #fix random seed for reproducibility
        np.random.seed(7)
        self.xtraining = []
        self.xtesting = []
        self.ytraining = []
        self.ytesting = []
        self.setx = []
        self.sety = []
        self.dataz = []
        self.t_steps = 3
        self.diff_step = 1
        self.axis = 0
        self.n_train = 2200
        self.n_test = 120
        self.setsToTrain = 1
        self.xymax = 800
        self.transformedData = []

    # Load Data
    def loadData(self,afile):
        outputName = afile[:afile.rindex('/') + 1] + 'xyrot' + afile[afile.rindex('/') + 1:]
        x_data = h5py.File(afile, 'r')
        x_data = x_data['matrices']
        y_data = h5py.File(outputName, 'r')
        y_data = y_data['matrices']
        y_data = y_data[:, :2]
        x_set, y_set = fn.sequence_input(x_data, y_data, self.t_steps, self.axis, self.diff_step)
        self.setx.append(x_set)
        self.sety.append(y_set)
        ### Reshape data
        print('Post-norm Shapes:')
        for x in range(len(self.setx)):
            print(self.setx[x])
            print(self.sety[x])
        self.setTraining()
        self.normalize(self.xymax)
    ### Select Amount of Training Examples
    def setTraining(self):
        for number in self.setsToTrain:
            self.xtraining.append(self.setx[number-1][:self.n_train,:,:,:,:])
            self.xtesting.append(self.setx[number-1][self.n_train:,:,:,:,:])
            self.ytraining.append(self.sety[number-1][:self.n_train, :, :, :, :])
            self.ytesting.append(self.sety[number-1][self.n_train:, :, :, :, :])
        for x in range(len(self.setx)):
            if x not in self.setsToTrain:
                self.setx[x] = self.setx[x][:self.n_test,:,:,:,:]
                self.sety[x] = self.sety[x][:self.n_test, :, :, :, :]

    #normailze Data
    def normalize(self,screenSize):
        xdata_mean = 0
        xdata_range = 0
        
        for i,xtrain in enumerate(self.xtraining):
            self.xtraining[i],xdata_mean,xdata_range = fn.data_normalize(xtrain, norm_type='mean', data_mean=None,data_range=None)
        for i,ytrain in enumerate(self.ytraining):
            self.ytraining[i] = fn.data_normalize(ytrain, norm_type='max', data_mean=0, data_range=screenSize)[0]
        for i,xtest in enumerate(self.xtesting):
            self.xtesting[i],xdata_mean,xdata_range = fn.data_normalize(xtest,norm_type='mean', data_mean=xdata_mean, data_range=xdata_range)
        for i,ytest in enumerate(self.ytesting):
            self.ytesting[i] = fn.data_normalize(ytest,norm_type='max', data_mean=0, data_range=screenSize)[0]
        for i,xset in enumerate(self.setx):
            self.setx[i],xdata_mean,xdata_range = fn.data_normalize(xset,norm_type='mean', data_mean=xdata_mean, data_range=xdata_range)
        for i,yset in enumerate(self.sety):
            self.sety[i] = fn.data_normalize(yset,norm_type='max', data_mean=0, data_range=screenSize)[0]
        
        print('Post-norm Shapes:')
        print(self.setx[4].shape)
        print(self.xtest[0].shape)
        print(self.xtraining[0].shape)
        print(self.sety[4].shape)
        print(self.ytesting[0].shape)
        print(self.ytraining[0].shape)

    def MAE(self,y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true) / K.abs(y_true))

    ##### CNN-LSTM Parameter Control Module #####
    def CNNLSTM(self,xtraining,
                xtesting,
                ytraining,
                ytesting,
                setx,
                sety,
                set_number = 0,
                n_unitsFC1 = 100,
                n_unitsFC2 = 100,
                LSTM_neurons = 100,
                n_convpool_layers = 1,
                n_convlayers = 3,
                n_reglayers = 1,
                max_poolsize = (3, 3),
                train_set_ratio = 0.7,
                valid_set_ratio = 0.15,
                drop_rate = 0.0,
                n_filters = 100,
                kernel_size = (3, 3),
                input_strides = 1,
                kernel_init = 'glorot_uniform',
                cost_function = 'mean_squared_error',
                batch_size = 10,
                n_epochs = 200,
                n_output = 2,
                n_post_test = 5000,
                early_stop_delta = 0.0005,  # 0.0025 change or above is considered improvement
                early_stop_patience = 10,  # keep optimizing for 10 iterations under "no improvement"
                out_filepath = 'trained_nets/file_001.h5'):
        
        print('Network Params:')

        print 'cost_function:', cost_function
        print 'n_unitsFC1:', n_unitsFC1
        print 'n_unitsFC2:', n_unitsFC2
        print 'LSTM_neurons:', LSTM_neurons
        print 'batch_size:', batch_size
        print 'kernel_size:', kernel_size
        print 'n_filters:', n_filters
        print 'max_poolsize:', max_poolsize
        print 'drop_rate:', drop_rate
        print 'early_stop_delta:', early_stop_delta
        print 'early_stop_patience:', early_stop_patience

        # We create a layer which take as input movies of shape
        # (n_frames, width, height, channels(RxGxBxXxY)) and returns the X,Y
        # coordinates of moving object

        self.model = Sequential()

        # define CNN self.model architecture

        self.model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=input_strides,
                                         kernel_initializer=kernel_init,
                                         activation='relu'), input_shape=(self.t_steps, 100, 100, 5)))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

        self.model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=input_strides,
                                         kernel_initializer=kernel_init,
                                         activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

        self.model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=input_strides,
                                         kernel_initializer=kernel_init,
                                         activation='relu')))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

        # self.model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
        #                  kernel_initializer= kernel_init,
        #                  activation='relu')))
        # self.model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

        self.model.add(TimeDistributed(Flatten()))

        self.model.add(TimeDistributed(Dense(n_unitsFC1, activation='relu', kernel_initializer=kernel_init)))
        self.model.add(TimeDistributed(Dense(n_unitsFC1, activation='relu', kernel_initializer=kernel_init)))
        # self.model.add(TimeDistributed(Dense(100, activation='relu')))
        # self.model.add(TimeDistributed(Dense(150, activation='relu')))
        self.model.add(TimeDistributed(Dense(10, activation='linear', kernel_initializer=kernel_init)))

        # Automatic Setup

        # for ii in range (n_convpool_layers):
        # for i in range(n_convlayers):
        #  self.model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size = kernel_size ,strides= input_strides,
        #                  kernel_initializer= kernel_init,
        #                  activation='relu'),input_shape=(3,100,100,5)))
        # self.model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

        # self.model.add(TimeDistributed(Flatten()))

        # define LSTM self.model

        # self.model.add(Bidirectional(CuDNNLSTM(LSTM_neurons, return_sequences=False,kernel_initializer= kernel_init)))
        # self.model.add(Bidirectional(CuDNNLSTM(LSTM_neurons, return_sequences=True,kernel_initializer= kernel_init)))
        self.model.add(CuDNNLSTM(LSTM_neurons, return_sequences=True, kernel_initializer=kernel_init))
        # self.model.add(CuDNNLSTM(LSTM_neurons, return_sequences=True,kernel_initializer= kernel_init))
        # self.model.add(CuDNNLSTM(LSTM_neurons,kernel_initializer= kernel_init))

        # for i in range(n_reglayers):
        # self.model.add(TimeDistributed(Dense(n_unitsFC2,activation='relu',kernel_initializer= kernel_init)))

        ## To use when outputing single time steps
        # self.model.add(Dense(n_unitsFC2,activation='relu',kernel_initializer= kernel_init))
        # self.model.add(Dense(n_unitsFC2,activation='relu',kernel_initializer= kernel_init))

        self.model.add(Dense(n_output, activation='linear', kernel_initializer=kernel_init))

        self.model.summary()



        self.model.compile(loss=cost_function, optimizer='adam', metrics=[self.MAE])

        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=early_stop_delta,
                                                      patience=early_stop_patience, verbose=1, mode='auto')

        # Train the network

        history = self.model.fit(xtraining[set_number], ytraining[set_number], epochs=n_epochs,
                            batch_size=batch_size, verbose=1,
                            callbacks=[earlyStopping],
                            validation_split=valid_set_ratio)
        #        validation_data =[inp_valid,out_valid])

        self.predict_initialTest()

        print 'cost_function:', cost_function
        print 'n_unitsFC1:', n_unitsFC1
        print 'n_unitsFC2:', n_unitsFC2
        print 'LSTM_neurons:', LSTM_neurons
        print 'batch_size:', batch_size
        print 'kernel_size:', kernel_size
        print 'n_filters:', n_filters
        print 'max_poolsize:', max_poolsize
        print 'drop_rate:', drop_rate
        print 'early_stop_delta:', early_stop_delta
        print 'early_stop_patience:', early_stop_patience
        # Save self.model
        # self.model.save(out_filepath)

        ## Post-Processing
    def predict_initialTest(self,fileNames):
        for i in range(0,len(self.xtraining)):
            print('Test Case ',str(i+1),':')
            print(fileNames[i+1])
            print('Actual Values')
            print(self.ytesting[i][110:112, :])
            print('Predicted Values')
            print((self.model.predict_on_batch(self.xtesting[i][110:112, :, :, :, :])))
            score = self.model.evaluate(self.xtesting[i], self.ytesting[i], verbose=0)
            print 'Test loss:', score[0]
            print 'Test error:', score[1]

    def predict_cases(self,fileNames, fileNumber):
        ## Post-Processing
        for i,case in self.setx:
            print('Test Case',str(i+2),':')
            print(fileNames[i+1])
            print('Actual Values')
            print(self.sety[i][110:112, :])
            print('Predicted Values')
            print((self.model.predict_on_batch(self.setx[i][110:112, :, :, :, :])))

            score = self.model.evaluate(self.setx[i], self.sety[i], verbose=0)
            print 'Test loss:', score[0]
            print 'Test error:', score[1]


        target_test = self.sety[0]
        output_test = self.model.predict_on_batch(self.setx[0])
        print(target_test.shape)
        print(output_test.shape)
        out = np.hstack((target_test, output_test))
        filename = '3ball_output'
        fileformat = '.csv'

        with open(filename + fileformat, 'wb') as csvfile:
            filewriter = csv.writer(csvfile)
            for j in range(0, out.shape[0]):
                filewriter.writerow(out[j, :])

        print('Network Params:')



        # Testing the network on one movie
        # feed it with the first 7 positions and then
        # predict the new positions

        # for j in range(16):
        #    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
        #    new = new_pos[::, -1, ::, ::, ::]
        #    track = np.concatenate((track, new), axis=0)

cnnlstm = CNNLSTM()
fileNames = ['datasets/borders/ball_in_polygon_long10ball.h5','datasets/borders/ball_in_square_long10ball.h5','datasets/long_projectile_motion/projectileMotion3sec120fps_100_300.h5',
             'datasets/3_ball/ball_in_polygon_longer3ball.h5','datasets/borders/ball_in_polygon20hits.h5','datasets/borders/ball_in_box20hits.h5','datasets/3_ball/ball_on_box400.h5']
for name in fileNames:
    cnnlstm.loadData(name)



    ### Select Amount of Training Examples


    ### Uncomment the following for single output pred
    ## also adjust net structure

    #y_train =  y_train [:,1,:]

    ### Normalize inputs and outputs





    #print(y_data[900:910,:])

    #quit()





