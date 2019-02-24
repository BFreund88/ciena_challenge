import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class dataset:
    def __init__(self,data,frac_train,frac_valid):
        # fix random seed for reproducibility
        seed = 42
        np.random.seed(seed)
        #Split the sample into training/validation/test
        y_full = data.iloc[:,0:12]
        x_full = data.iloc[:,12:]

        #Normalize input variables
        scaler = MinMaxScaler()
        print(scaler.fit(x_full))
        x_full = scaler.transform(x_full)

        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(x_full, y_full,test_size=0.2, random_state=seed)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train_full, self.y_train_full,test_size=0.2, random_state=seed)

        #Save shape of the train validation and test sample
        self.shape_train = self.X_train.shape
        self.shape_valid = self.X_valid.shape
        self.shape_test = self.X_test.shape

def prepare_data(frac_train,frac_valid):
    #Read csv files and prepare them as panda data frames
    data = pd.read_csv('TrainingDataset.csv') 
    #Remove NaN values from dataset
    data = data.fillna(-1.)
    #Store the data set in the dataset class
    data_set = dataset(data,frac_train,frac_valid)
    return data_set

def plot_loss(logs, val_logs, title):
    plt.plot(logs, label='train')
    plt.plot(val_logs, label='valid')
    plt.legend()
    plt.title('Training loss')
    plt.savefig('./OutputPlots/'+title+'.png')
    plt.clf()
