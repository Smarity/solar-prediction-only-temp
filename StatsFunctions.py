'''
Import libraries and functions
'''
import math
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sqrt = math.sqrt
mean = statistics.mean

def get_mean_bias_error(x, y):
    '''
    It calculates the mean bias error function (MBE).

    Input:
        * x -> list or array of measured/actual data. For example,
        in terms of et0, it would be FAO56-PM et0.

        * y -> list or array of predicted values. For example, in
        terms of et0, it would be the predicted et0 by a neural
        network model, etc.

    Output:
        * mbe -> a float with the mean bias error value
    '''
    # check the lengths are the same 
    assert len(x)==len(y)
    # convert the inputs in numpy array
    x = np.array(x)
    y = np.array(y)
    # mbe calculus
    delta = y - x
    mbe = float(sum(delta)*1.0/len(x))
    return(mbe)

def get_root_mean_square_error(x, y):
    '''
    It calculates de root mean square error (RMSE).

    Input:
        * x -> list or array of measured/actual data. For example,
        in terms of et0, it would be FAO56-PM et0.

        * y -> list or array of predicted values. For example, in
        terms of et0, it would be the predicted et0 by a neural
        network model, etc.

    Output:
        * rmse -> a float with the root mean squared error value
    '''
    # check the lengths are the same 
    assert len(x)==len(y)
    # convert the inputs in numpy array
    x = np.array(x)
    y = np.array(y)
    # rmse calculus
    delta = (y - x)**2
    rmse = sqrt(sum(delta)*1.0/len(x))
    return rmse

def get_relative_error(x, y):
    '''
    It calculates de relative error RE. 

    Input:
        * x -> list or array of measured/actual data. For example,
        in terms of et0, it would be FAO56-PM et0.

        * y -> list or array of predicted values. For example, in
        terms of et0, it would be the predicted et0 by a neural
        network model, etc.

    Output:
        * re -> a float with the relative error value. The value given 
        is a function of per-unit, instead of per-cent.
    '''
    # check the lengths are the same 
    assert len(x)==len(y)
    # convert the inputs in numpy array
    x = np.array(x)
    y = np.array(y)
    # re calculus
    mbe = get_mean_bias_error(x, y)
    x_mean = mean(x)
    re = mbe/x_mean
    return re

def get_ratio_error(x, y):
    '''
    It calculates the ratio between both average estimated and measured

    Input:
        * x -> list or array of measured/actual data. For example,
        in terms of et0, it would be FAO56-PM et0.

        * y -> list or array of predicted values. For example, in
        terms of et0, it would be the predicted et0 by a neural
        network model, etc.

    Output:
        * re -> a float with the relative error value. The value given 
        is a function of per-unit, instead of per-cent.
    '''
    # check the lengths are the same 
    assert len(x)==len(y)
    # convert the inputs in numpy array
    x = np.array(x)
    y = np.array(y)
    # ratio error calculus
    return mean(y)/mean(x)

def get_coefficient_of_determination(x, y):
    '''
    It calculates the coefficient of determination (R**2)

    Input:
        * x -> list or array of measured/actual data. For example,
        in terms of et0, it would be FAO56-PM et0.

        * y -> list or array of predicted values. For example, in
        terms of et0, it would be the predicted et0 by a neural
        network model, etc.
    Output:
        * R2 -> a float with the coefficient of determination in 
        per unit, instead of percentage.
    '''
    # check the lengths are the same 
    assert len(x)==len(y)
    # convert the inputs in numpy array
    x = np.array(x)
    y = np.array(y)

    # determines the means of the values
    x_mean = mean(x)
    y_mean = mean(y)
    delta_measured = x - x_mean
    delta_measured_2 = (x - x_mean)**2
    delta_prediction = y - y_mean
    delta_prediction_2 = (y - y_mean)**2
    R2 = (sum(delta_measured*delta_prediction)/sqrt(sum(delta_measured_2)*sum(delta_prediction_2)))**2
    return R2

def get_nash_suteliffe_efficiency(x, y):
    '''
    It calculates NSE. It is calculated by sklearn.metrics.r2_score

    Input:
        * x -> list or array of measured/actual data. For example,
        in terms of et0, it would be FAO56-PM et0.

        * y -> list or array of predicted values. For example, in
        terms of et0, it would be the predicted et0 by a neural
        network model, etc.
    Output:
        * nse -> a float with the coefficient of determination in 
        per unit, instead of percentage.
    '''
    # check the lengths are the same 
    assert len(x)==len(y)
    # convert the inputs in numpy array
    x = np.array(x)
    y = np.array(y)

    return r2_score(x, y)


