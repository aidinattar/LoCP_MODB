import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from tsfresh import extract_features
from xgboost import XGBClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import seaborn as sns
from sklearn import metrics

def get_df(x):
    '''Build input dataframe for given data series
    Input:
    var = array of time series, (#samples,time,1)
    Return:
    df = dataframe ready for features extraction
    '''

    #N = #samples, t = timesteps
    N, t = x.shape[0], x.shape[1]
    #build id columns
    id_col = np.repeat(np.arange(N),t)
    #build time columns
    time_col = np.tile(np.arange(t),N)
    #build var columns
    x_col = x.flatten()

    #build dict for df
    x_dict = {'id':id_col,'time':time_col,'value':x_col}

    #return dataframe
    return pd.DataFrame(x_dict)

# function for the random step, using lambda construction
# int() for cleaner look and for mimicing a detector with finite resolution
jump = lambda drift, stdev: int(np.random.normal(drift,stdev))
def pattern(i,z,a):
    return int(a*np.sin((np.pi*i)/z))

    # pattern parameters: Z=nr of steps, A=amplitude
Z=12
A=500

# number of samples to be used
dim = [20, 50, 100, 150, 200, 250, 300, 400, 500]

# size of each sample of the timeseries
L=60
# step parameters: introduce small positive bias 
DX = 50
bias = 5

iterations = 20
train_accuracy =     val_accuracy     = np.zeros(shape=(len(dim), iterations))
train_accuracy_CNN = val_accuracy_CNN = np.zeros(shape=(len(dim), iterations))

for z in range(iterations):
    x = np.array([np.zeros(shape=(i, 60)) for i in range(len(dim))])
    y = np.array([np.zeros(shape=(i    )) for i in range(len(dim))])
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
    df = [get_df(x[i]) for i in range(len(dim))]
    x_features = [extract_features(df[i], #our dataframe
                               column_id='id', #sample id, from 0 to N
                               column_sort='time', #timestep, from 0 to t
                               column_kind=None, #we have only one feature
                               column_value='value', #value of input 
                               n_jobs=4) #number of cores
                  for i in range(len(dim))]
    #remove columns with NaN or inf
    for i in range(len(dim)):
        x_features[i].replace([np.inf, -np.inf], np.nan)
        x_features[i] = x_features[i].dropna(axis='columns')
    #split data into training and validation

    perc_train=0.8
    N_train=[int(perc_train*len(x[i]))  for i in range(len(dim))]
    x_train=[x_features[i][:N_train[i]] for i in range(len(dim))]
    y_train=[y[i][:N_train[i]]          for i in range(len(dim))]
    x_val  =[x_features[i][N_train[i]:] for i in range(len(dim))]
    y_val  =[y[i][N_train[i]:]          for i in range(len(dim))]
    N_val  =[len(x_val[i])              for i in range(len(dim))]


    for i in range(len(dim)):
        x_train[i].drop(columns=['value__sample_entropy'],inplace=True)
        x_val[i].drop(columns=['value__sample_entropy'],inplace=True)

    # reproducibility
    np.random.seed(12345)

    #define parameters for xgboost
    params = {'max_depth':6,'min_child_weight':1,\
            'learning_rate':0.3,'use_label_encoder':False}

    #build model with given params
    model = [XGBClassifier(**params) for i in range(len(dim))]
    scaler = StandardScaler()
    x_trainScaled = [scaler.fit_transform(x_train[i]) for i in range(len(dim))]

    #fit
    for i in range(len(dim)):
        model[i].fit(x_train[i].values,y_train[i])

    # plot tree here if needed

    #predict labels on training set
    y_pred_train = [model[i].predict(x_train[i]) for i in range(len(dim))]
    #predict labels on validation set
    y_pred_val = [model[i].predict(x_val[i]) for i in range(len(dim))]
    y_pred_val_soft = [model[i].predict_proba(x_val[i]) for i in range(len(dim))]

    #compute accuracies
    train_accuracy[:, z] = [accuracy_score(y_train[i],y_pred_train[i]) for i in range(len(dim))]
    val_accuracy[:, z]   = [accuracy_score(y_val[i],y_pred_val[i])     for i in range(len(dim))]

results_fits = pd.DataFrame({'validation':  np.concatenate([train_accuracy[i,:]                for i in range(len(dim))]),
                             'training':    np.concatenate([val_accuracy[i,:]                  for i in range(len(dim))]),
                             'group':       np.concatenate([np.repeat(f'{dim[i]}', iterations) for i in range(len(dim))])})
results_fits.to_csv('XGB_3.csv')
