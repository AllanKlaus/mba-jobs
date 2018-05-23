
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import cross_val_predict
from matplotlib import pyplot as plot


import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
import sklearn.cross_validation


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
plot.style.use('fivethirtyeight')
plot.rcParams['figure.figsize'] = (15,5)


# In[18]:


boston_houses = load_boston()
boston_houses_table = pd.DataFrame(boston_houses.data)
boston_houses_table.columns = boston_houses.feature_names
boston_houses_table['PRICE'] = boston_houses.target

linear_regression = LinearRegression()


# In[24]:


def cross_validation_with_prediction(linear_regression, data):
    y = data.target

    predicted = cross_val_predict(linear_regression, data.data, y, cv=10)

    fig, ax = plot.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plot.show()
    
    return predicted

prediction = cross_validation_with_prediction(linear_regression, boston_houses)


# In[28]:


def quality_metrics(data, linear_regression, prediction):
    x = data.data
    y = data.target

    print('Coefficients: ', linear_regression.coef_)
    print("MSE: %.2f" % mean_squared_error(y, prediction))
    print("MAE: %.2f" % mean_absolute_error(y, prediction))
    print("MdAE: %.2f" % median_absolute_error(y, prediction))

quality_metrics(boston_houses, linear_regression, prediction)


# In[25]:


def cross_validation_with_prediction_only_train_values(linear_regression, data):
    train_quantity = 350
    
    x_train = data.data[0:train_quantity + 1]
    #x_test  = data.data[train_quantity + 1:]
    
    y_train = data.target[0:train_quantity + 1]
    #y_test  = data.target[train_quantity + 1:]

    predicted = cross_val_predict(linear_regression, x_train, y_train, cv=10)

    fig, ax = plot.subplots()
    ax.scatter(y_train, predicted, edgecolors=(0, 0, 0))
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plot.show()
    
    return predicted

train_prediction = cross_validation_with_prediction_only_train_values(linear_regression, boston_houses)


# In[31]:


def quality_metrics_only_train_values(data, linear_regression, prediction):
    train_quantity = 350
    
    x = data.data[0:train_quantity + 1]
    y = data.target[0:train_quantity + 1]
    
    linear_regression.fit(x, y)

    print("MSE: %.2f" % mean_squared_error(y, prediction))
    print("MAE: %.2f" % mean_absolute_error(y, prediction))
    print("MdAE: %.2f" % median_absolute_error(y, prediction))

quality_metrics_only_train_values(boston_houses, linear_regression, train_prediction)

