import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2, l1_l2
#from tensorflow.keras.metrics import MeanAbsolutePercentageError

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import csv
import os  
from functools import wraps
import time

#import kmeans_initializer ### Cloned from Github Repo !git clone https://github.com/PetraVidnerova/rbf_for_tf2
#import rbflayer
### Calcultes how much time a function needed to complete
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end-start)
        return result
    return wrapper
def csv_writer(path,lists):
  with open(path+'/iterative.csv','w',newline='') as f:
    writer = csv.writer(f)
    for row in zip(*lists):
      writer.writerow(row)
    print("written succesfully in "+ path)
    
    
 
    
    
    
def Styblinski_Tang_Function(x, y) :
    return 0.5 * (x**4 + y**4) - 8 * (x**2 + y**2) + 2.5 * (x + y)
def stf(n_sample):
  X = np.random.uniform(-5,5,(n_sample,2))
  y = Styblinski_Tang_Function(X[:,0], X[:,1])
  return X, y 
def f1(n_sample, d=6):
  X = np.random.uniform(0,10,(n_sample,d))
  y = 1/2 * X[:,0]**2 + X[:,1]**2+(1/np.math.factorial(3))*X[:,2]**3 \
  -(1/np.math.factorial(4))*X[:,3]**2 + X[:,4] - (X[:,5]-3)\
  -5*X[:,2]+5
  return X,y 

f,target_f1 = f1(1000)
plt.figure(figsize=(12,3))
plt.plot(target_f1,'go',fillstyle='none')
plt.ylabel('$f_1$')
plt.show()
