import numpy as np
import matplotlib.pyplot as plt
import keras,sklearn
import tensorflow as tf
import pandas as pd
import ast
import seaborn as sns


from keras.models import Sequential
from keras.layers import Dropout, Dense, MaxPool1D, Conv1D
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

plt.rcParams['font.size'] = 14

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def filename(s,TYPE=1):
    return "./DATA/"+s+"-for-DNN-"+str(TYPE)+".dat"

#training data
perc_train = 0.8

#keras works with numpy arrays: just use them from the start
TYPE = 1
x = np.loadtxt(filename('data', TYPE), delimiter=' ')
y = np.loadtxt(filename('labels', TYPE), delimiter=' ')
N = len(x)

x_red = np.loadtxt(filename('data_red', TYPE), delimiter=' ')
y_red = np.loadtxt(filename('labels_red', TYPE), delimiter=' ')
N_red = len(x_red)

x_inc = np.loadtxt(filename('data_inc', TYPE), delimiter=' ')
y_inc = np.loadtxt(filename('labels_inc', TYPE), delimiter=' ')
N_inc = len(x_inc)


#dim of a sample
L = len(x[0])
print('Regular set:', L)

L_red = len(x_red[0])
print('Reduced set:', L_red)

L_inc = len(x_inc[0])
print('Increased set:', L_inc)


N_train_red = int( perc_train * N_red )
N_train_inc = int( perc_train * N_inc )
N_train     = int( perc_train * N     )
print(f'Regular data\t\t: {N}\t\t\ttrain\t\t: {N_train}')
print(f'Reduced data\t\t: {N_red}\t\t\ttrain\t\t: {N_train_red}')
print(f'Increased data\t\t: {N_inc}\t\t\ttrain\t\t: {N_train_inc}')

(x_train, y_train) = (x[:N_train],y[:N_train])
(x_valid, y_valid) = (x[N_train:],y[N_train:])
print("Train:",len(x_train),"\t Validation:",len(x_valid))

(x_train_red, y_train_red) = (x_red[:N_train_red],y_red[:N_train_red])
(x_valid_red, y_valid_red) = (x_red[N_train_red:],y_red[N_train_red:])
print("Train:",len(x_train_red),"\t Validation:",len(x_valid_red))

(x_train_inc, y_train_inc) = (x_inc[:N_train_inc],y_inc[:N_train_inc])
(x_valid_inc, y_valid_inc) = (x_inc[N_train_inc:],y_inc[N_train_inc:])
print("Train:",len(x_train_inc),"\t Validation:",len(x_valid_inc))

x_train_aug = np.zeros( shape = (x_train.shape[0] * 10, x_train.shape[1]) )
y_train_aug = np.zeros( shape =  y_train.shape[0] * 10 )
for i in range( x_train.shape[0] ):
    S    = np.random.normal( 0, 1 , size = ( 10, 2 ) )
    x_train_aug[i*10:i*10+10] = np.array( [x[i, 0] + S[:, 0], x[i, 1] + S[:, 1]] ).T
    y_train_aug[i*10:i*10+10] = y[i]
x_valid_aug, y_valid_aug = x_valid, y_valid

x_aug = np.concatenate( [x_train_aug, x_valid_aug] )
y_aug = np.concatenate( [y_train_aug, y_valid_aug] )

N_aug = len(x_aug)

N_train_aug = len(x_train_aug)
print(f'Augmented data\t\t: {N_aug}\t\t\ttrain\t\t: {N_train_aug}')

L_aug = len(x_aug[0])
print('Augmented set:', L_inc)

print("Train:",len(x_train_aug),"\t Validation:",len(x_valid_aug))


def Rescale(x):
    #return (x-x.mean())/np.sqrt(x.var())
    return x/50

x_train     = Rescale(x_train)
x_valid     = Rescale(x_valid)

x_train_red = Rescale(x_train_red)
x_valid_red = Rescale(x_valid_red)

x_train_inc = Rescale(x_train_inc)
x_valid_inc = Rescale(x_valid_inc)

x_train_aug = Rescale(x_train_aug)
x_valid_aug = Rescale(x_valid_aug)


x_red_red = np.loadtxt(filename('data_red_red', TYPE), delimiter=' ')
y_red_red = np.loadtxt(filename('labels_red_red', TYPE), delimiter=' ')
N_red_red = len(x_red_red)


L_red_red = len(x_red_red[0])
print('Reduced set:', L_red_red)


N_train_red_red = int( perc_train * N_red_red )

print(f'Reduced data\t\t: {N_red_red}\t\t\ttrain\t\t: {N_train_red_red}')

(x_train_red_red, y_train_red_red) = (x_red_red[:N_train_red_red],y_red_red[:N_train_red_red])
(x_valid_red_red, y_valid_red_red) = (x_red_red[N_train_red_red:],y_red_red[N_train_red_red:])
print("Train:",len(x_train_red_red),"\t Validation:",len(x_valid_red_red))

x_train_red_red = Rescale(x_train_red_red)
x_valid_red_red = Rescale(x_valid_red_red)


x_inc_inc = np.loadtxt(filename('data_inc_inc', TYPE), delimiter=' ')
y_inc_inc = np.loadtxt(filename('labels_inc_inc', TYPE), delimiter=' ')
N_inc_inc = len(x_inc_inc)


L_inc_inc = len(x_inc_inc[0])
print('incuced set:', L_inc_inc)


N_train_inc_inc = int( perc_train * N_inc_inc )

print(f'incuced data\t\t: {N_inc_inc}\t\t\ttrain\t\t: {N_train_inc_inc}')

(x_train_inc_inc, y_train_inc_inc) = (x_inc_inc[:N_train_inc_inc],y_inc_inc[:N_train_inc_inc])
(x_valid_inc_inc, y_valid_inc_inc) = (x_inc_inc[N_train_inc_inc:],y_inc_inc[N_train_inc_inc:])
print("Train:",len(x_train_inc_inc),"\t Validation:",len(x_valid_inc_inc))

x_train_inc_inc = Rescale(x_train_inc_inc)
x_valid_inc_inc = Rescale(x_valid_inc_inc)

nepoch = 400
def create_DNN(activation, dropout_rate, layers):
    model = Sequential()
    model.add(Dense(L,input_shape=(L,),activation = activation))
    for i in range(len(layers)):
        model.add(Dense(layers[i],activation = activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation="sigmoid"))
    return model

def compile_model(optimizer=tf.keras.optimizers.Adam(), activation = "relu", dropout_rate = 0.2, layers = (20, 20)):
    # create the mode
    model=create_DNN(activation, dropout_rate, layers)
    # compile the model
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

hist_red_red = []
hist_red     = []
hist         = []
hist_inc     = []
hist_inc_inc = []
hist_aug     = []

hist_red_red_acc = []
hist_red_acc     = []
hist_acc         = []
hist_inc_acc     = []
hist_inc_inc_acc = []
hist_aug_acc     = []

hist_red_red_loss = []
hist_red_loss     = []
hist_loss         = []
hist_inc_loss     = []
hist_inc_inc_loss = []
hist_aug_loss     = []

hist_red_red_loss_val = []
hist_red_loss_val     = []
hist_loss_val         = []
hist_inc_loss_val     = []
hist_inc_inc_loss_val = []
hist_aug_loss_val     = []
n = 10

for i in range(10):
    model_red_red = compile_model()
    fit = model_red_red.fit(x_train_red_red, y_train_red_red,
                            epochs=nepoch, batch_size=50,
                            validation_data=(x_valid_red_red,y_valid_red_red),
                            verbose=0)
    hist_red_red_acc.append(fit.history['accuracy'][-1])
    hist_red_red.append(fit.history['val_accuracy'][-1])
    hist_red_red_loss.append(fit.history['loss'][-1])
    hist_red_red_loss_val.append(fit.history['val_loss'][-1])
    print(i, 'red_red')
    del fit, model_red_red

    model_red = compile_model()
    fit = model_red.fit(x_train_red, y_train_red,
                            epochs=nepoch, batch_size=50,
                            validation_data=(x_valid_red,y_valid_red),
                            verbose=0)
    hist_red_acc.append(fit.history['accuracy'][-1])
    hist_red.append(fit.history['val_accuracy'][-1])
    hist_red_loss.append(fit.history['loss'][-1])
    hist_red_loss_val.append(fit.history['val_loss'][-1])
    print(i, 'red')
    del fit, model_red

    model = compile_model()
    fit = model.fit(x_train, y_train,
                            epochs=nepoch, batch_size=50,
                            validation_data=(x_valid,y_valid),
                            verbose=0)
    hist_acc.append(fit.history['accuracy'][-1])
    hist.append(fit.history['val_accuracy'][-1])
    hist_loss.append(fit.history['loss'][-1])
    hist_loss_val.append(fit.history['val_loss'][-1])
    print(i, 'reg')
    del fit, model

    model_inc = compile_model()
    fit = model_inc.fit(x_train_inc, y_train_inc,
                            epochs=nepoch, batch_size=50,
                            validation_data=(x_valid_inc,y_valid_inc),
                            verbose=0)
    hist_inc_acc.append(fit.history['accuracy'][-1])
    hist_inc.append(fit.history['val_accuracy'][-1])
    hist_inc_loss.append(fit.history['loss'][-1])
    hist_inc_loss_val.append(fit.history['val_loss'][-1])
    print(i, 'inc')
    del fit, model_inc

    model_inc_inc = compile_model()
    fit = model_inc_inc.fit(x_train_inc_inc, y_train_inc_inc,
                            epochs=nepoch, batch_size=50,
                            validation_data=(x_valid_inc_inc,y_valid_inc_inc),
                            verbose=0)
    hist_inc_inc_acc.append(fit.history['accuracy'][-1])
    hist_inc_inc.append(fit.history['val_accuracy'][-1])
    hist_inc_inc_loss.append(fit.history['loss'][-1])
    hist_inc_inc_loss_val.append(fit.history['val_loss'][-1])
    print(i, 'inc_inc')
    del fit, model_inc_inc

    model_aug = compile_model()
    fit = model_aug.fit(x_train_aug, y_train_aug,
                            epochs=nepoch, batch_size=50,
                            validation_data=(x_valid_aug,y_valid_aug),
                            verbose=0)
    hist_aug_acc.append(fit.history['accuracy'][-1])
    hist_aug.append(fit.history['val_accuracy'][-1])
    hist_aug_loss.append(fit.history['loss'][-1])
    hist_aug_loss_val.append(fit.history['val_loss'][-1])
    print(i, 'aug')
    del fit, model_aug

results_fits = pd.DataFrame({'validation':  np.concatenate([hist_red_red, hist_red, hist, hist_inc, hist_inc_inc, hist_aug]),
                             'training':  np.concatenate([hist_red_red_acc, hist_red_acc, hist_acc, hist_inc_acc, hist_inc_inc_acc, hist_aug_acc]),
                             'loss_val':  np.concatenate([hist_red_red_loss_val, hist_red_loss_val, hist_loss_val, hist_inc_loss_val, hist_inc_inc_loss_val, hist_aug_loss_val]),
                             'loss':  np.concatenate([hist_red_red_loss, hist_red_loss, hist_loss, hist_inc_loss, hist_inc_inc_loss, hist_aug_loss]),
                             'group': np.concatenate([np.repeat('500 samples', n),np.repeat('2000 samples', n),np.repeat('4000 samples', n),np.repeat('8000 samples', n),np.repeat('16000 samples', n),np.repeat('Augmented set', n)])})
results_fits.to_csv('Boxplot_7.csv')
