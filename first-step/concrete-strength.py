
# this follows the same idea as german-credit.py
# but instead of classification, this is a regression problem
# again, this data is from the uci machine learning repository


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.neural_network import MLPRegressor




# i really wish there was a python package like 'here' in R
#  b/c this way of specifying a file path is annoying
# i didn't find anything after a quick google search but maybe i'll look into
#  it later

file_path = "/Users/andrewbates/Desktop/neural-nets/data/concrete-strength.csv"

concrete = pd.read_csv(file_path, sep = ",")

concrete.shape
concrete.head()


# really should do some plots to check out the data
# that might be added later but the point of this exercise is to get 
#  familiar with modeling in python so it will have to wait

# ------- set up train/test and scale inputs --------

x = concrete.iloc[:,0:8]
y = concrete.iloc[:,8]

x.head()
y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42)


scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# make sure scaling done
x.head()
x_train 


# ------ elastic net ---------

# just use default settings
enet = ElasticNet()
enet.fit(x_train, y_train)
enet_pred = enet.predict(x_test)

# let's just use MSE for simplicity
metrics.mean_squared_error(y_test, enet_pred) # 135.37


# ---- neural network ------

# most of these parameters are based on the provided example file
# not exactly sure about some of them yet but i'm sure we'll get into
#  tuning later on

# for now, let's use stochastic gradient descent
# it's supposed to not need as many iterations 
# to investigate this, we'll fit a few models with different # of max iterations

# 100 iterations max
mlp100 = MLPRegressor(hidden_layer_sizes = (10,10),
                      activation = "logistic",
                      solver = "sgd",
                      learning_rate = "adaptive",
                      max_iter = 100,
                      learning_rate_init = 0.01,
                      early_stopping = True,
                      validation_fraction = 0.1,
                      alpha = 0.01)


# 250 iterations max
# note: the only difference from above is the max_iter argument
mlp250 = MLPRegressor(hidden_layer_sizes = (10,10),
                      activation = "logistic",
                      solver = "sgd",
                      learning_rate = "adaptive",
                      max_iter = 250,
                      learning_rate_init = 0.01,
                      early_stopping = True,
                      validation_fraction = 0.1,
                      alpha = 0.01)

# 500 iterations max
mlp500 = MLPRegressor(hidden_layer_sizes = (10,10),
                      activation = "logistic",
                      solver = "sgd",
                      learning_rate = "adaptive",
                      max_iter = 500,
                      learning_rate_init = 0.01,
                      early_stopping = True,
                      validation_fraction = 0.1,
                      alpha = 0.01)

# 1,000 iterations max. this is what was in the provided file
mlp1k = MLPRegressor(hidden_layer_sizes = (10,10),
                      activation = "logistic",
                      solver = "sgd",
                      learning_rate = "adaptive",
                      max_iter = 1000,
                      learning_rate_init = 0.01,
                      early_stopping = True,
                      validation_fraction = 0.1,
                      alpha = 0.01)


# now fit them all
mlp100.fit(x_train, y_train)
mlp250.fit(x_train, y_train)
mlp500.fit(x_train, y_train)
mlp1k.fit(x_train, y_train)


# and get predictions
mlp100_pred = mlp100.predict(x_test)
mlp250_pred = mlp250.predict(x_test)
mlp500_pred = mlp500.predict(x_test)
mlp1k_pred = mlp1k.predict(x_test)


# and finally get the MSE
metrics.mean_squared_error(y_test, mlp100_pred) # 39.01
metrics.mean_squared_error(y_test, mlp250_pred) # 33.99
metrics.mean_squared_error(y_test, mlp500_pred) # 39.77
metrics.mean_squared_error(y_test, mlp1k_pred) # 45.78

# and the winner is ... 250 iterations max

# the MSE for the elastic net was 135.37
( (135.37 - 33.99) / 135.37 ) * 100.0
# that's a 75% increase in MSE!








