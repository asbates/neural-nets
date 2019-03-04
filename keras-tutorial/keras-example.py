

# this is a basic example of how to use keras

# we use a version of the ames housing data set
# http://jse.amstat.org/v19n3/decock.pdf
# this version was adapted from the R package AmesHousing
# only the numeric predictors are included
# each predictor is centerd and scaled
# the outcome is log transformed

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

file_path = "/Users/andrewbates/Desktop/neural-nets/data/ames-numeric.csv"
ames = pd.read_csv(file_path)

ames.shape
ames.columns

# create x,y and train/test split
ames_x = ames.drop('Log_Sale_Price', axis = 1)
ames_y = ames['Log_Sale_Price']

p = len(ames_x.columns)

x_train, x_test, y_train, y_test = train_test_split(ames_x, 
                                                    ames_y,
                                                    test_size = 0.25)


# initialize a sequential model
# this is the base model from which we build upon
model = Sequential()

# add 2 hidden layers and one output layer
model.add(Dense(units = 10, activation = 'sigmoid', input_dim = p))
model.add(Dense(units = 10, activation = 'sigmoid'))
model.add(Dense(units = 1))

# set things up for training
model.compile(loss = 'mse', optimizer = 'sgd', metrics = ['mse'])

# train the model
model.fit(x_train, y_train, epochs = 10)

# make predictions
pred = model.predict(x_test)

metrics.mean_squared_error(y_test, pred)













