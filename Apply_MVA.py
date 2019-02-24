import argparse
import sys
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras import optimizers
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np
import pandas as pd
import matplotlib as mpl
#To allow running in batch mode
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from os import listdir
from os.path import isfile, join
from common_functions import *

"""Evaluation of best performing MVA model on test set 

Usage:  
  python Apply_NN.py --model [--v] [--printout]

Required Argument
  --model=<model> Model that is to be applied on the test sample (BDT or NN)

Options: 
  -h --help             Show this screen. 
Optional arguments
  --v=<verbose> Set verbose level
  --printout=<printout> Prints out a few predictions and true sales values
  --bagging=<bagging> Average predictions with several NNs
"""
def feature_ranking_BDT(model,shape):
    importances = model.estimators_[0].feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(shape):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #Save feature ranking in text file
    np.savetxt('Ranking.txt', np.c_[indices,importances[indices]],fmt="%i %f")

def print_prediction(prediction, y_test):
    #Print a few examples of the prediction
    print('Here a comparison of predictions and true sale values of the test set')
    print('Prediction of sales values')
    print(prediction)
    print('Compared to the true values')
    print(data_set.y_test)

def load_best_model(path,model):
    #Load best performing model
    listfiles = [f for f in listdir(path) if isfile(join(path, f))]
    listfiles.sort()
    if(len(listfiles)>1):
        if model=='NN': 
            model = load_model(path+'/'+listfiles[1])
            model.summary()
        elif model=='BDT': 
            model = joblib.load(path+'/'+listfiles[1])
            print(model)
        return model
    else:
        raise Exception('Directory empty')

def load_models_NN(path,bagging):
    #Load several best performing NN models
    listfiles = [f for f in listdir(path) if isfile(join(path, f))]
    listfiles.sort()
    models = []
    if(listfiles>bagging):
        for i in range(bagging):
            models.append(load_model(path+'/'+listfiles[i+1]))
        return models
    else:
        raise Exception('Not enough models available')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'MVA evaluation on Test set')
    parser.add_argument("--v", "--verbose", help="increase output verbosity", default=True, type=bool)
    parser.add_argument('--model', help = "Decide between BDT and NN application", type=str, required=True)
    parser.add_argument('--printout', help = "Print out predictions and true sales values", action='store_true')
    parser.add_argument('--bagging', help = "Average predictions over several NN", default=True, type=int)

    args = parser.parse_args()
    print('The following model will be applied on the testing set:')
    print(args)

    #Load data set
    #Set fractions for testing and validation
    if (args.v): print('Read Data Set')
    frac_train=0.8
    frac_valid=0.8
    data_set=prepare_data(frac_train,frac_valid)

    #Load best performing model
    if (args.v): print('Load {} Model'.format(args.model))

    #Calculate test error for BDT
    if (args.model == 'BDT'):
        path = 'OutputBest_BDT'
        if (args.v): print('Loading Model from directory {}'.format(path))
        model = load_best_model(path,'BDT')

        #Calculate error on test set and compare to validation error
        pred_val = model.predict(data_set.X_valid)
        scores_valid = [mean_squared_error(data_set.y_valid, pred_val, multioutput='uniform_average'), mean_absolute_error(data_set.y_valid, pred_val, multioutput='uniform_average')]
        pred_test = model.predict(data_set.X_test)
        scores_test = [mean_squared_error(data_set.y_test, pred_test, multioutput='uniform_average'), mean_absolute_error(data_set.y_test, pred_test, multioutput='uniform_average')]
        print('The error on the validation set with the best performing NN is (mean square and absolute error): {}'.format(scores_valid))
        print('The error on the Test set with the best performing BDT is: {}'.format(scores_test))

        #Print out predictions and true labels
        if (args.printout): print_prediction(pred_test[:10,:],data_set.y_test.iloc[:10,:])

        #Rank the input features in importance
        feature_ranking_BDT(model,data_set.X_test.shape[1])

    #Calculate test error for NN
    elif (args.model == 'NN'):
        path = 'OutputBest'
        if (args.v): print('Loading Model from directory {}'.format(path))
        model = load_best_model(path,'NN')
        #Calculate error on the validation and test set
        if (args.v): print('Calculate Test score')
        scores_valid = model.evaluate(data_set.X_valid,data_set.y_valid,verbose=args.v)
        scores_test = model.evaluate(data_set.X_test,data_set.y_test,verbose=args.v)
        print('The error on the validation set with the best performing NN is: {}'.format(scores_valid))
        print('The error on the Test set with the best performing NN is: {}'.format(scores_test))

        #Print out predictions and true labels
        if (args.printout): print_prediction(model.predict(data_set.X_test)[:10,:],data_set.y_test.iloc[:10,:])

        #Average predictions of several NN
        print('Average NN')
        if(args.bagging>0): 
            models  = load_models_NN(path,args.bagging)
            predictions = np.zeros((data_set.X_test.shape[0],data_set.y_train.shape[1]))
            for i in models:
                predictions+=i.predict(data_set.X_test)
            predictions/=len(models)
            errors = [mean_squared_error(data_set.y_test, predictions),mean_absolute_error(data_set.y_test, predictions)]

            print('The error on the Test set with an average over {} best performing NNs is: {}'.format(args.bagging,errors))

    else:
        raise Exception('Model has to be either NN or BDT')
