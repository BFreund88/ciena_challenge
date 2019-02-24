import argparse
import sys
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib as mpl
#To allow running in batch mode
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from common_functions import *

#BDT model as implemented in sklearn
#AdaBoost
def BDTada(max_depth, learning_rate, n_estimators):
    BDTada= MultiOutputRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth),
                              learning_rate=learning_rate,
                              n_estimators=n_estimators))
    return BDTada

#Gradien Boost
def BDTModelgrad(max_depth, learning_rate, n_estimators,verbose,n_iter_no_change):
    BDTgrad = MultiOutputRegressor(GradientBoostingRegressor(max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        n_estimators=n_estimators,
                                        verbose=verbose,
                                        n_iter_no_change=n_iter_no_change))
    return BDTgrad

"""Boosted Decision Tree Optimisation   

Usage:  
  python Train_BDT.py

Options: 
  -h --help             Show this screen. 
Optional arguments 
  --output =<output>    Specify output name
  --depth=<depth> Maximal depth of estimators in Ensemble 
  --nest=<nest> Number of estimators in Ensemble 
  --early=<early> Early stopping for Gradient Boosting Regressor

  --v=<verbose> Set Verbose level
  --lr=<lr> Learning rate for SGD optimizer
  --opt=<opt> Chose between Ensemble Methods AdaBoost and Gradient Boosting Regressor (0 or 1)
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'BDT optimisation')
    parser.add_argument("--v", "--verbose", help="increase output verbosity", default=0, type=int)
    parser.add_argument("--output", help="Specify Output name", default='', type=str)
    parser.add_argument('--depth', help = "Specifies the maximal depth of trees in Ensemble", default=3, type=int)
    parser.add_argument('--nest', help = "Specifies the number of estimators in Ensemble", default=200, type=int)
    parser.add_argument('--lr','--learning_rate', help = "Specifies the learning rate", default=0.01, type=float)
    parser.add_argument('--early', help = "Specifies the condition for early stopping for Gradient Boosting Regressor", default=0, type=int)
    parser.add_argument('--opt', help = "Specifies the optimizer used: AdaBoost:0 and Gradient Boost:1 (default),", default=1, type=int)

    args = parser.parse_args()
    print('Train with hyper parameters:')
    print(args)

    #Read data files
    #Set fractions for testing and validation
    print('Read Data Set')
    frac_train=0.8
    frac_valid=0.8
    data_set=prepare_data(frac_train,frac_valid)

    #Print input dimensions   
    print('Input Shape: {}'.format(data_set.shape_train))

    #Show number of events in each category
    num_train=data_set.shape_train[0]
    num_valid=data_set.shape_valid[0]
    num_test=data_set.shape_test[0]                           

    num_tot=num_train+num_valid+num_test    

    print('Number of training: {}, validation: {}, test: {} and total events: {}.'.format(num_train,num_valid,num_test,num_tot))

    #Define Ensemble models with given hyper parameters
    modelada=BDTada(args.depth,args.lr,args.nest)
    modelboost=BDTModelgrad(args.depth,args.lr,args.nest,args.v,args.early)

    opt = [modelada,modelboost]

    print(opt)

    print("Fit Model")
    print(data_set.X_train.shape,data_set.y_train.values.shape)
    opt[args.opt].fit(data_set.X_train, data_set.y_train.values)

    #Evaluate performance on validation set

    #Prediction
    pred_val = opt[args.opt].predict(data_set.X_valid)
    #Calculate mean absolute and squared error on validation set
    absolute_error = mean_absolute_error(data_set.y_valid, pred_val, multioutput='uniform_average')
    square_error = mean_squared_error(data_set.y_valid, pred_val, multioutput='uniform_average')
    print('Mean squared: {} and absolute error: {}'.format(np.round(square_error), np.round(absolute_error)))

    #Save model with mean absolute error in name to select best model
    print("Save Model")
    filenameBDT = './OutputBest_BDT/'+str(int(np.round(absolute_error)))+'_'+args.output+'modelBDT_train.pkl'
    _ = joblib.dump(opt[args.opt], filenameBDT, compress=9)
