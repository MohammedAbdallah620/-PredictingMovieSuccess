import pandas as pd
import json

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import Lasso
from DataCleaner import *
import os.path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import pickle
from sklearn.decomposition import PCA

import  time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

def regression(X_train,X_test,y_train,y_test,features):

    poly_features = PolynomialFeatures(degree=1)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)
    # poly Regression
    poly_triningTime, poly_testingTime, poly_accuracy = 0, 0, 0
    filename = 'poly_reg.sav'
    if os.path.exists(filename):
        model_poly = pickle.load(open(filename, 'rb'))
    else:
        t1 = time.time()
        model_poly = linear_model.LinearRegression().fit(X_train_poly, y_train)
        t2 = time.time()
        pickle.dump(model_poly, open(filename, 'wb'))
        poly_triningTime = t2 - t1
    t1 = time.time()
    poly_predicted = model_poly.predict(X_test_poly)
    poly_score = model_poly.score(X_test_poly, y_test)
    t2 = time.time()
    poly_testingTime = t2 - t1
    poly_accuracy = poly_score * 100

    # lin Regression
    linear_triningTime, linear_testingTime, linear_accuracy = 0, 0, 0
    filename = 'linear_reg.sav'
    if os.path.exists(filename):
        model_linear = pickle.load(open(filename, 'rb'))
    else:
        t1 = time.time()
        model_linear = linear_model.LinearRegression().fit(X_train, y_train)
        t2 = time.time()
        linear_triningTime = t2 - t1
        pickle.dump(model_linear, open(filename, 'wb'))
    t1 = time.time()
    prediction = model_linear.predict(X_test)
    linear_score = model_linear.score(X_test, y_test)
    t2 = time.time()
    linear_testingTime = t2 - t1
    linear_accuracy = linear_score * 100

    ##Ridge Regression
    ridge_triningTime, ridge_testingTime, ridge_accuracy = 0, 0, 0
    filename = 'ridge_reg.sav'
    if os.path.exists(filename):
        model_ridge = pickle.load(open(filename, 'rb'))
    else:
        t1 = time.time()
        model_ridge = Ridge(alpha=0.01).fit(X_train,
                                            y_train)  # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, co
        t2 = time.time()
        ridge_triningTime = t2 - t1
        pickle.dump(model_ridge, open(filename, 'wb'))
    t1 = time.time()
    Ridge_test_score = model_ridge.score(X_test, y_test)
    pred = model_ridge.predict(X_test)
    t2 = time.time()
    ridge_testingTime = t2 - t1
    mse_ridge = np.mean((pred - y_test) ** 2)
    ridge_accuracy = Ridge_test_score * 100

    # Lasso Regression
    lasso_triningTime, lasso_testingTime, lasso_accuracy = 0, 0, 0
    filename = 'lasso_reg.sav'
    if os.path.exists(filename):
        model_lasso = pickle.load(open(filename, 'rb'))
    else:
        t1 = time.time()
        model_lasso = Lasso(alpha=0.3).fit(X_train, y_train)
        t2 = time.time()
        lasso_triningTime = t2 - t1
        pickle.dump(model_lasso, open(filename, 'wb'))
    t1 = time.time()
    pred_lasso = model_lasso.predict(X_test)
    lasso_score = model_lasso.score(X_test, y_test)
    t2 = time.time()
    lasso_testingTime = t2 - t1
    mse_lasso = np.mean((pred_lasso - y_test) ** 2)
    lasso_accuracy = lasso_score * 100

    # bar graphs
    model_name = ('linear', 'poly', 'lasso', 'ridge')
    y_pos = np.arange(len(model_name))
    training_time = [linear_triningTime, poly_triningTime, lasso_triningTime, ridge_triningTime]
    testing_time = [linear_testingTime, poly_testingTime, lasso_testingTime, ridge_testingTime]
    accuracy = [linear_accuracy, poly_accuracy, lasso_accuracy, ridge_accuracy]

    plt.bar(y_pos, training_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Training Time')
    plt.show()

    plt.bar(y_pos, testing_time, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('time')
    plt.title('Testing Time')
    plt.show()

    plt.bar(y_pos, accuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, model_name)
    plt.ylabel('accuracy')
    plt.title('Acuuracy')
    plt.show()
    #############


    '''for i in features:
        plt.scatter(X_train[i], y_train)

        X1 = np.min(X_test[i])
        X1List = np.where(X_test[i] == X1)[0]
        X1Index = X1List[0]
        Y1 = prediction[X1Index]

        X2 = np.max(X_test[i])
        X2List = np.where(X_test[i] == X2)[0]
        X2Index = X2List[0]
        Y2 = prediction[X2Index]
        XX = [X1, X2]
        YY = [Y1, Y2]
        plt.xlabel(i)
        plt.ylabel('rate')
        plt.plot(XX, YY, 'red')

        # plt.show()
        '''
    print("okk")
    print('Mean Square Error Of Linear', metrics.mean_squared_error(y_test, prediction))
    print('Score Of Linear', model_linear.score(X_test, y_test))
    print('--------------------------')
    print('Mean Square Error Of Ploy', metrics.mean_squared_error(y_test, poly_predicted))
    print('Score Of Ploy', poly_score)
    print('--------------------------')
    print('Mean Square Error Of Ridge Regression ', mse_ridge)
    print('Score Of Ridge Regression', Ridge_test_score)
    print('--------------------------')
    print('Mean Square Error Of Lasso Regression ', mse_lasso)
    print('Score Of Lasso Regression', lasso_score)


