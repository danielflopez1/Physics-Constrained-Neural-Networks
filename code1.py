import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.io import wavfile
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, LSTM, MaxPooling2D, CuDNNLSTM, Bidirectional
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
import glob, os
from pandas.tools.plotting import table
import random
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")
last4Digits = 3981
X = pd.read_csv("data/Xrs.csv", sep=',', header=None, dtype=float)
X = X.values
#something = X[900:1300].to_csv("data/Xrs.csv", sep=',',header = None,index=False)
yb = pd.read_csv("data/ybrs.csv",header = None)
yb = yb.values.ravel()#[900:1300]
#something = yb[900:1300].to_csv("data/ybrs.csv", sep=',',header = None,index=False)
print(X)
print(yb)


grids = []
for imag in X:
    grids.append(imag.reshape((64,64,1)))
grids = np.array(grids)
print(grids.shape)


X_train, X_test, y_train, y_test = train_test_split(grids, yb, test_size=1./3, random_state=last4Digits)
accuracies = []
numbers = []
def modelcreation(a,b,c,d):
    print("-------",a,"----",b,"------",c)
    model = Sequential()

    #convolutional layer with rectified linear unit activation
    model.add(Conv2D(c, kernel_size=(a, a),input_shape=(64,64,1), activation='sigmoid'))
    #32 convolution filters used each of size 3x3
    #again
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #64 convolution filters used each of size 3x3
    #choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(b, b)))
    #randomly turn neurons on and off to improve convergence
    #model.add(Dropout(0.25))
    #flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    #fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    #one more dropout for convergence' sake :)
    #model.add(Dropout(0.5))
    #output a softmax to squash the matrix into output probabilities
    model.add(Dense(1, activation='relu'))

    batch_size = 128
    num_epoch = d
    #model training
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model_log = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=num_epoch,
              verbose=0,
              validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0]) #Test loss: 0.0296396646054
    print('Test accuracy:', score[1]) #Test accuracy: 0.9904
    accuracies.append(score[1])
    numbers.append((a,b,c,d))
    print(accuracies)
    print(numbers)

for a in range(1,6):
    for b in range(1,6):
        for c in range(32,257,32):
            for d in range(1,10):
                modelcreation(a, b, c,d)
'''

def modstring(strat):
    inx = 0.0
    nums = []
    inx = strat.find("\'test_f1\'")
    arrx = strat.find("[", inx)
    arrend = strat.find("]", arrx)
    nums.append(strat[arrx:arrend + 1])
    avgsnums = []
    # print(nums)
    for num in nums:
        num = num.replace("[", "")
        num = num.replace("]", "")
        num = num.replace(" ", "")
        num = num.split(',')
        addition = 0.0
        for numb in num:
            addition += float(numb)
        avgsnums.append(addition / 3.0)
    # print(avgsnums)
    return avgsnums
compo = 100

MaxesMax = 0
X = pd.read_csv("data/X.csv", sep=' ', header=None, dtype=float)
X = X.values#[900:1300] #start 1s 1100 to 1629    200 0s 100 1s
pca =  PCA(n_components=60,random_state=last4Digits, whiten=True,svd_solver='randomized').fit(X)
X = pca.transform(X)
print(X.shape)
yb = pd.read_csv("data/y_bush_vs_others.csv",header = None)
yb = yb.values.ravel()#[900:1300]

yw = pd.read_csv("data/y_williams_vs_others.csv",header = None)
#ys = y.values.ravel()
yw = yw.values.ravel()#[9900:10001]

num_jobs = 1
print(yw.shape)
print(yb.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
'''