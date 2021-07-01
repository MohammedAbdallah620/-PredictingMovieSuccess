import pandas as pd
import json

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from DataCleaner import *
import os.path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

if not os.path.exists('numerical_data.xlsx'):
    runDataCleaner()

movies = pd.read_excel('numerical_data.xlsx')

for key in ['budget', 'runtime', 'revenue','genres','keywords','production_companies','production_countries','spoken_languages','cast','crew']:
    mean = movies[key].describe()['mean']
    movies.loc[(movies[key] == 0) | (movies[key].isnull()), key] = mean


X=movies.iloc[:,1:13] #Features


Y=movies['vote_average']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)

poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
model = linear_model.LinearRegression()
#poly Train
model.fit(X_train_poly, y_train)
poly_predicted = model.predict(X_test_poly)
poly_score=model.score(X_test_poly, y_test)
#lin Train
model.fit(X_train, y_train)
prediction = model.predict(X_test)
# predicting on training data-set

print('Mean Square Error Of Linear', metrics.mean_squared_error(y_test, prediction))
print('Score Of Linear', model.score(X_test, y_test))
print('Mean Square Error Of Ploy', metrics.mean_squared_error(y_test, poly_predicted))
print('Score Of Ploy',poly_score )

##Ridge Regression
rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely
rr.fit(X_train, y_train)
#Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
pred = rr.predict(X_test)
mse_rr = np.mean((pred - y_test)**2)

#Lasso Regression
lassoReg = Lasso(alpha=0.3)
lassoReg.fit(X_train,y_train)
pred_lasso= lassoReg.predict(X_test)
mse_lasso = np.mean((pred_lasso - y_test)**2)
lasso_score=lassoReg.score(X_test,y_test)

print('Mean Square Error Of Ridge Regression ', mse_rr)
print('Score Of Ridge Regression', Ridge_test_score)
print('Mean Square Error Of Lasso Regression ', mse_lasso)
print('Score Of Lasso Regression', lasso_score)