import argparse
import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib as mpl
#To allow running in batch mode
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from common_functions import *

#Keras model with number of hidden layers, number of neurons and dropout renormalization as hyper parameters
def KerasModel(input_dim,numlayer,numn, dropout):
    model = Sequential()
    model.add(Dense(numn, input_dim=int(input_dim)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    for i in range(numlayer-1):
        model.add(Dense(numn))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
    model.add(Dense(12))
    return model


"""Neural Network Optimisation   

Usage:  
  python Train_NN.py

Options: 
  -h --help             Show this screen. 
Optional arguments 
  --output =<output>    Specify output name
  --numlayer=<numlayer>     Specify number of hidden layers.
  --numn=<numn> Number of neurons per hidden layer 
  --dropout=<dropout> Dropout to reduce overfitting 
  --epoch=<epochs> Specify training epochs
  --lr=<lr> Learning rate for SGD optimizer
  --patience=<patience> Set patience for early stopping
  --opt=<opt> Chose between Adadelta and Adam optimizer (0 or 1)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'NN optimisation')
    parser.add_argument("--v", "--verbose", help="increase output verbosity", default=0, type=int)
    parser.add_argument("--output", help="Specify Output name", default='', type=str)
    parser.add_argument('--numlayer', help = "Specifies the number of layers of the Neural Network", default=2, type=int)
    parser.add_argument('--numn', help = "Specifies the number of neurons per hidden layer", default=20, type=int)
    parser.add_argument('--lr','--learning_rate', help = "Specifies the learning rate for SGD optimizer", default=0.01, type=float)
    parser.add_argument('--dropout', help = "Specifies the applied dropout", default=0.05, type=float)
    parser.add_argument('--epochs', help = "Specifies the number of epochs", default=80, type=int)
    parser.add_argument('--patience', help = "Specifies the patience for early stopping", default=5, type=int)
    parser.add_argument('--opt', help = "Specifies the optimizer used: Adadelta:0 and Adam:1 (default),", default=1, type=int)

    args = parser.parse_args()
    print('Train with hyper parameters:')
    print(args)

    #Read data files
    #Set fractions for testing and validation
    print('Read Data Set')
    frac_train=0.8
    frac_valid=0.8
    data_set=prepare_data(frac_train,frac_valid)

    #Get input dimensions   
    shape_train=data_set.X_train.shape
    print('Input Shape: {}'.format(shape_train))
    shape_valid=data_set.X_valid.shape
    shape_test=data_set.X_test.shape 

    num_train=shape_train[0]
    num_valid=shape_valid[0]
    num_test=shape_test[0]                           

    num_tot=num_train+num_valid+num_test    

    print('Number of training: {}, validation: {}, test: {} and total events: {}.'.format(num_train,num_valid,num_test,num_tot))

    #Define Keras NN model with given hyper parameters

    model=KerasModel(shape_train[1],args.numlayer,args.numn,args.dropout)

    #Possible optimizers                               
    sgd = optimizers.SGD(lr=args.lr, decay=1e-6, momentum=0.6, nesterov=True)
    ada= optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.01)
    nadam=optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    opt=[ada,nadam]

    model.compile(loss='mean_squared_error', optimizer=opt[args.opt], metrics=['msle','mae'])
    model.summary()
    model.save('./OutputModel/'+args.output+'modelNN_initial.h5')

    #Define checkpoint to save best performing NN
    #Use early stopping with patience
    path='./OutputModel/'+args.output+'output_NN.h5'
    print('Saving as {}'.format(path))
    callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience),ModelCheckpoint(filepath=path, monitor='val_loss', verbose=args.v, save_best_only=True)]

    #Train Model
    print('Fit Model')
    logs = model.fit(data_set.X_train, data_set.y_train.values, epochs=args.epochs,
                     validation_data=(data_set.X_valid, data_set.y_valid),batch_size=20, callbacks=callbacks, verbose =args.v)

    #Plot loss as a number of epochs
    # list all data in history
    print(logs.history.keys())
    plot_loss(logs.history['loss'],logs.history['val_loss'],args.output+'loss')
    plot_loss(logs.history['mean_absolute_error'],logs.history['val_mean_absolute_error'],args.output+'absolute_error')

    #Evaluate performance on validation set
    
    #Restores best performing Model and compiles automatically    
    model = load_model('./OutputModel/'+args.output+'output_NN.h5')
    
    #Evaluate model
    print('Evaluate')
    scores = model.evaluate(data_set.X_valid, data_set.y_valid, verbose=args.v)
    print(scores)

    #Save model in OutputBest with the total error in the name to identify best performing NN
    model.save('./OutputBest/'+str(int(np.round(scores[2])))+'_'+args.output+'best_NN.h5')
