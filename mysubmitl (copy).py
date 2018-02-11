import numpy as np 
import pandas as pd 
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
import csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier #For Classification
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
#from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn import linear_model
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('final.csv') #usecols = head12
df2 = pd.read_csv('newcorrect.csv') #usecols = head12
y1 = df.RESULT
y2 = np.column_stack( y1 )
y3 = y2.T
y3 = y3.ravel()
#y = y2.reshape( (1,len(y2) ) )
#print(y)

X1 = [df.TILT_ANGLE,	df.INITIAL_VELOCITY,	df.FINAL_VELOCITY, df.ACCLERATION, df.TILT_THRESHOLD,		df.SPEED_LIMIT,	df.sudden_acceleration,	df.SUDDEN_STOP,	df.EXCESS_BIKE_TILT]
X = np.column_stack( X1 )


X2 =  [ df2.TILT_ANGLE,	df2.INITIAL_VELOCITY,	df2.FINAL_VELOCITY,	df2.ACCLERATION,	df2.TILT_THRESHOLD,	df2.SPEED_LIMIT,	df2.sudden_acceleration,	df2.SUDDEN_STOP,	df2.EXCESS_BIKE_TILT ]
#X1, X2, y3, y0 = train_test_split(X1, y3, test_size=0.3, random_state=0)

X = np.column_stack( X1 )
Xl = np.column_stack( X2 )
#print(X.shape)
#print(y3.shape)
#clf = AdaBoostClassifier()
#clf1 =  GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
#reg = SGDRegressor(penalty='elasticnet', alpha=0.01,l1_ratio=0.25, fit_intercept=True)
#reg = reg.fit(X,y3)
#r = reg.predict(Xl)

#a = linear_model.SGDRegressor()
#a.fit(X, y3)
#q = a.predict(Xl)

#params = {'n_estimators': 1000, 'max_depth': 8, 'min_samples_split': 2,
     #     'learning_rate': 0.02, 'loss': 'ls'}
#d = ensemble.GradientBoostingRegressor(**params)

#d.fit(X, y3)
#mse = mean_squared_error(y_test, clf.predict(X_test))
#test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

#d = d.predict(Xl)

clf1 = XGBClassifier()
clf1.fit(X,y3)
#clf = svm.SVC()
#clf4 = tree.DecisionTreeClassifier()			#decision tree						                #simple SVM
#clf1 = GaussianNB() 					          #naive_bayes					
#clf3 = GaussianProcessClassifier() 
#clf = clf.fit(X,y3)
#h = clf4.fit(X,y3)
#clf1 = clf1.fit(X,y3)

#c = clf.predict(Xl)
c1 = clf1.predict(Xl)
#l = clf4.predict(X)
#c = c.ravel()
#c = c.T
#r =r.T
#q= q.T
#d= d.T
df3 = pd.DataFrame({"Output" :c1,  "TIME" : df2.TIME_OF_RECORDING})
df3.to_csv("output112.csv", index=False)


#d= d.reshape(d.shape[0],1)
#y3= y3.reshape(y3.shape[0],1)
#print(y3.shape)
#print(d.shape)
#print("gbr")
#print(accuracy_score(y0,c1 , normalize=True) )
#print("xg:")
#print(accuracy_score(y3, c1, normalize=True) )
