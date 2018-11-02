# -*- coding: utf-8 -*-


import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("C:/Users/vmoha/Desktop/Collection/Fire/forestfires.csv",   header=0)
print(dataset.describe())
print(dataset.isnull().sum())
all_features= dataset[['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH',
       'wind', 'rain']]
all_classes= dataset.area
print(dataset.area[dataset.area == 0].count()/dataset.area.count())



#### Preprocessing Step #######
#..............Transformg month and day features into numbers

    
def month_transform(X):
    X.loc[X['month'] == 'jan', 'month'] = 1
    X.loc[X['month'] == 'feb', 'month'] = 2
    X.loc[X['month'] == 'mar', 'month'] = 3
    X.loc[X['month'] == 'apr', 'month'] = 4
    X.loc[X['month'] == 'may', 'month'] = 5
    X.loc[X['month'] == 'jun', 'month'] = 6
    X.loc[X['month'] == 'jul', 'month'] = 7
    X.loc[X['month'] == 'aug', 'month'] = 8
    X.loc[X['month'] == 'sep', 'month'] = 9
    X.loc[X['month'] == 'oct', 'month'] = 10
    X.loc[X['month'] == 'nov', 'month'] = 11
    X.loc[X['month'] == 'dec', 'month'] = 12
    return X

def day_transform(X):
    X.loc[X['day'] == 'mon', 'day'] = 1
    X.loc[X['day'] == 'tue', 'day'] = 2
    X.loc[X['day'] == 'wed', 'day'] = 3
    X.loc[X['day'] == 'thu', 'day'] = 4
    X.loc[X['day'] == 'fri', 'day'] = 5
    X.loc[X['day'] == 'sat', 'day'] = 6
    X.loc[X['day'] == 'sun', 'day'] = 7
    return X
    
    
    

all_features = month_transform(all_features)
all_features = day_transform(all_features)


# Scatter Plot matrix #######
from pandas.plotting import scatter_matrix
scatter_matrix(all_features, alpha=0.2, figsize=(15, 15), diagonal='kde')

#.............................Scaling matric .............................
print(dataset.corr())
#.............................................Scaling the features . ....................
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(all_features)
all_features_scaled = scaler.transform(all_features)




#...... ..Splitting the data int traing and test data = 25%

X_train, X_test, y_train, y_test = train_test_split(all_features_scaled, all_classes, test_size=0.25)


################  Ridge regression with built-in cross-validation ############

clf = RidgeCV(alphas=[1e-3, 2* 1e-3, 3* 1e-1, 4* 1e-1,1, 2, 3, 4, 5]).fit(X_train , y_train)
print(clf.score(X_train , y_train)) 
y_test_perdict = clf.predict(X_test)

###### plotting the data ##############3
import numpy as np
import matplotlib.pyplot as plt


##... plotting the training data versus tge perdicted data
# red dashes, blue squares and green triangles
plt.figure()
t = np.arange(0, X_train.shape[0])
plt.plot(t, clf.predict(X_train), 'rs', t, y_train, 'bs')
plt.show()


##### The score is too low, we will try poly nomial

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

cv_Score_list=[]
for n in [1,2,3, 4, 5, 6, 7, 8]:
    poly = PolynomialFeatures(degree=n)
    X_poly = poly.fit_transform(X_train)
    RidgeCV(alphas=[1e-3, 2* 1e-3, 3* 1e-1, 4* 1e-1,1, 2, 3, 4, 5])
    clf2 = linear_model.LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(all_features_scaled, all_classes, test_size=0.25)
    cv_scores = cross_val_score(clf2, X_poly, y_train, cv =5, scoring='r2')
    print("cv_Score by Polynomial Model for n = %d is:" %n, cv_scores.mean())
    cv_Score_list.append(cv_scores.mean())
    
fig = plt.figure()
plt.title('Negative Mean Square Error for different polynomial degrees')
plt.xlabel("Degree of polynomial(n)")
plt.ylabel("Negative Mean Square Error")
plt.plot([1,2,3, 4, 5, 6, 7, 8], cv_Score_list,'r*')
# fig.savefig('D:/37-Oil_projct/polynomical degree Error.png')
plt.show()

# As you see even with polynomical the r2 score is a negative value. This shoews that the model acts poorly. We should try more complicated models.


