from tensorflow.keras.utils import to_categorical

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten #, Reshape
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D
import pandas as pd

from keras import initializers, regularizers
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

from scipy import stats
import seaborn
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from keras import initializers, regularizers
from statistics import mean
import tensorflow.random as tf_r

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from xgboost import XGBClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import seaborn as sns
from sklearn import metrics

def reg_scale(x):
    return x/400

def std_scale(x, N):
    xm = x.mean(axis=1)
    for i in range(N):
        x[i] = x[i]-xm[i]
    return x/np.sqrt(x.var())

def norm_scale(x):
    return (x - np.min(x, axis=0)) / np.abs(np.max(x, axis=0) - np.min(x, axis=0))

def log_scale(x):
    x = x - np.min(x, axis=0) + 0.0000001 #shifting the data to be greater than 0
    return np.log10(x)

def create_CNN(reg, lam, init, kernel_size, input_shape):
    model = Sequential()
    model.add(Conv1D(filters=6, kernel_size=kernel_size[0],
                    kernel_initializer=init,
                    kernel_regularizer=reg(lam),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv1D(filters=8, kernel_size=kernel_size[1],
                    kernel_initializer=init,
                    kernel_regularizer=reg(lam),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv1D(filters=4, kernel_size=kernel_size[2], 
                    kernel_initializer=init, 
                    kernel_regularizer=reg(lam),
                    activation='relu', 
                    input_shape=input_shape)) 

    model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, 
    optimizer = "adam", metrics = ["accuracy"])

    return model

from statistics import mean

# function for the random step, using lambda construction
# int() for cleaner look and for mimicing a detector with finite resolution
jump = lambda drift, stdev: int(np.random.normal(drift,stdev))
def pattern(i,z,a):
    return int(a*np.sin((np.pi*i)/z))

# number of samples to be used
dim = [20, 50, 100, 150, 200, 250, 300, 400, 500]
    # pattern parameters: Z=nr of steps, A=amplitude
Z=12
A=500
# step parameters: introduce small positive bias 
DX = 50
bias = 5

iterations = 50
train_accuracy = np.zeros(shape=(len(dim), iterations))
val_accuracy   = np.zeros(shape=(len(dim), iterations))
train_accuracy_CNN = val_accuracy_CNN = np.zeros(shape=(len(dim), iterations))

init_best        = 'GlorotUniform' #is GlorotUniform
reg_best         = regularizers.L1 #best_result.param_reg.values[0]
lam_best         = 0
kernel_size_best = [5,5,5] #best_result.param_kernel_size.values[0]

BATCH_SIZE = 5
EPOCHS = 100
dim = [20, 50, 100, 150, 200, 250, 300, 400, 500]


for z in range(iterations):
    x = np.array([np.zeros(shape=(i, 60)) for i in range(len(dim))], dtype=object)
    y = np.array([np.zeros(shape=(i    )) for i in range(len(dim))], dtype=object)

    # size of each sample of the timeseries
    L=60

    # number of data samples
    for k, N in enumerate(dim):
        y[k] = [0] * N
        x[k] = np.array([[0] * L for i in range(N)])
        for i in range(N):
            if i>0:
                x[k][i][0] = x[k][i-1][-1] + jump(bias,DX)
            for j in range(1,L):
                x[k][i][j] = x[k][i][j-1] + jump(bias,DX)
            y[k][i] = i%3
            if y[k][i]>0:
                j0 = np.random.randint(0,L-1-Z)
                sign = 3-2*y[k][i]
                for j in range(Z):
                    x[k][i][j0+j] += sign*pattern(j,Z,A)

    N = [len(x[i])    for i in range(len(dim))]
    L = [len(x[i][0]) for i in range(len(dim))]

    x_CNN     =  [std_scale( x[i], N[i]) for i in range(len(dim))]

    perc_train=0.8
    N_train_CNN=[int(perc_train*len(x[i]))                for i in range(len(dim))]
    x_train_CNN=[x[i][:N_train_CNN[i]]                    for i in range(len(dim))]
    y_train_CNN=[to_categorical(y[i][:N_train_CNN[i]], 3) for i in range(len(dim))]
    x_val_CNN=[x[i][N_train_CNN[i]:]                      for i in range(len(dim))]
    y_val_CNN=[to_categorical(y[i][N_train_CNN[i]:],3)    for i in range(len(dim))]
    N_val_CNN=[len(x_val_CNN[i])                              for i in range(len(dim))]

    # Keras wants an additional dimension with a 1 at the end
    for i in range(len(dim)):
        x_train_CNN[i] = x_train_CNN[i].reshape(x_train_CNN[i].shape[0], L[i], 1)
        x_val_CNN[i] =  x_val_CNN[i].reshape(x_val_CNN[i].shape[0], L[i], 1)

    input_shape = [(L[i], 1) for i in range(len(dim))]

    model_best = [create_CNN(reg_best, lam_best, init_best, kernel_size_best, input_shape[i]) for i in range(len(dim))]

    fit_CNN = [model_best[i].fit(x_train_CNN[i],y_train_CNN[i],
                                epochs=EPOCHS,batch_size=BATCH_SIZE,
                                validation_data=(x_val_CNN[i], y_val_CNN[i]),
                                verbose=0, shuffle=True) for i in range(len(dim))]

    #compute accuracies
    train_accuracy[:, z] = [mean(fit_CNN[i].history['accuracy'][-10:-1]) for i in range(len(dim))]
    val_accuracy[:, z]   = [mean(fit_CNN[i].history['val_accuracy'][-10:-1]) for i in range(len(dim))]

results_fits = pd.DataFrame({'validation':  np.concatenate([train_accuracy[i,:]                for i in range(len(dim))]),
                             'training':    np.concatenate([val_accuracy[i,:]                  for i in range(len(dim))]),
                             'group':       np.concatenate([np.repeat(f'{dim[i]}', iterations) for i in range(len(dim))])})
results_fits.to_csv('CNN_6.csv')
