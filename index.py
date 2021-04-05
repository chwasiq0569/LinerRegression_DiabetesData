import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


diabetes = datasets.load_diabetes()
# dict_keys(['data', 'target', 'frame', 'DESCR',
#           'feature_names', 'data_filename', 'target_filename'])

diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-30]  # statring 30
diabetes_X_test = diabetes_X[-20:]  # last 30

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-20:]


model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_Y_train)  # training model
diabetes_Y_predicted = model.predict(diabetes_X_test)  # testing model
