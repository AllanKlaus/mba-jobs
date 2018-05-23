
# coding: utf-8

# In[50]:


from sklearn.datasets import load_boston
from sklearn import datasets, linear_model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import ShuffleSplit
from matplotlib import pyplot as plot
from sklearn.linear_model import LinearRegression


# In[64]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
import sklearn.model_selection


# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
plot.style.use('fivethirtyeight')
plot.rcParams['figure.figsize'] = (15,5)


# In[55]:


boston_houses = load_boston()


# In[56]:


print(boston_houses.keys())
print(boston_houses.data.shape)
print(boston_houses.feature_names)
print(boston_houses.DESCR)


# In[57]:


boston_houses_table = pd.DataFrame(boston_houses.data)
boston_houses_table.columns = boston_houses.feature_names
boston_houses_table['PRICE'] = boston_houses.target
print(boston_houses_table.head())


# In[69]:


def linear_regression(data):

    #get only one feature (in this case number of rooms)
    data_x = data.data[:, np.newaxis, 5]

    qtd_train = 350

    #split data in train and test
    x_train = data_x[0:qtd_train + 1]
    x_test  = data_x[qtd_train + 1:]

    y_train = data.target[0:qtd_train + 1]
    y_test  = data.target[qtd_train + 1:]
    
    #creating model
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    y_predictions = regr.predict(x_test)

    print('Coefficients: ', regr.coef_)

    print("MSE: %.2f" % mean_squared_error(y_test, y_predictions))
    print("MAE: %.2f" % mean_absolute_error(y_test, y_predictions))
    print("MdAE: %.2f" % median_absolute_error(y_test, y_predictions))
    return x_test, y_test, y_predictions

x_test, y_test, y_predictions = linear_regression(boston_houses)


# In[60]:


plot.scatter(x_test, y_test,  color='black')
plot.plot(x_test, y_predictions, color='blue', linewidth=3)
plot.xlabel("QTD MEDIA DOS QUARTOS")
plot.ylabel("PRECO MEDIO DAS CASAS")

plot.xticks(())
plot.yticks(())

plot.show()


# In[85]:


def cross_validation(data, column):
    x = data.drop(column, axis = 1)
    y = data[column]
    return sklearn.cross_validation.train_test_split(x, y, test_size = 0.33, random_state = 5)

x_train, x_test, y_train, y_test = cross_validation(boston_houses_table, 'PRICE')


# In[86]:


def graphic_linear_regression(x_train, x_test, y_train, y_test):
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    y_prediction = lm.predict(x_test)

    plot.scatter(y_test, y_prediction)
    plot.xlabel("Prices: $y_i$")
    plot.ylabel("Predicted prices: $\hat{Y}_i$")
    plot.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
    
graphic_linear_regression(x_train, x_test, y_train, y_test)

