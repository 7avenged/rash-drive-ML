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
df2 = pd.read_csv('newcorrect2.csv') #usecols = head12
y1 = df.RESULT
y2 = np.column_stack( y1 )
y3 = y2.T
y3 = y3.ravel()

y1b = df.SPEED_LIMIT
y2b = np.column_stack( y1b )
y3b = y2b.T
y3b = y3b.ravel()

y1c = df.sudden_acceleration
y2c = np.column_stack( y1c )
y3c = y2c.T
y3c = y3c.ravel()

y1d = df.SUDDEN_STOP
y2d = np.column_stack( y1d )
y3d = y2d.T
y3d = y3d.ravel()

y1e = df.EXCESS_BIKE_TILT
y2e = np.column_stack( y1e )
y3e = y2e.T
y3e = y3e.ravel()


y0 = df2.RESULT
y0 = y0.T
y0 = y0.ravel()

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
clf2 = XGBClassifier()
clf3 = XGBClassifier()
clf4 = XGBClassifier()
clf5 = XGBClassifier()
clf1.fit(X,y3)
clf2.fit(X,y3b)
clf3.fit(X,y3c)
clf4.fit(X,y3d)
clf5.fit(X,y3e)


#clf = svm.SVC()
#clf4 = tree.DecisionTreeClassifier()			#decision tree						                #simple SVM
#clf1 = GaussianNB() 					          #naive_bayes					
#clf3 = GaussianProcessClassifier() 
#clf = clf.fit(X,y3)
#h = clf4.fit(X,y3)
#clf1 = clf1.fit(X,y3)

#c = clf.predict(Xl)

c1 = clf1.predict(Xl)
c2 = clf1.predict(Xl)
c3 = clf1.predict(Xl)
c4 = clf1.predict(Xl)
c5 = clf1.predict(Xl)




#l = clf4.predict(X)
#c = c.ravel()
#c = c.T
#r =r.T
#q= q.T
#d= d.T
#df3 = pd.DataFrame({"Output" :c1,  "TIME-OR" : df2.TIME_OF_RECORDING})
#df3.to_csv("output113a.csv", index=False)

#df3 = pd.DataFrame({"Output" :c2,  "TIME-SPEED_LIMIT" : df2.TIME_OF_RECORDING})
#df3.to_csv("output113b.csv", index=False)

#df3 = pd.DataFrame({"Output" :c3,  "TIME-SUD-ACCL" : df2.TIME_OF_RECORDING})
#df3.to_csv("output113c.csv", index=False)

#df3 = pd.DataFrame({"Output" :c4,  "TIME-SUD-STOP" : df2.TIME_OF_RECORDING})
#df3.to_csv("output113d.csv", index=False)

#df3 = pd.DataFrame({"Output" :c5,  "TIME-EXCESS_TILT" : df2.TIME_OF_RECORDING})
#df3.to_csv("output113e.csv", index=False)

#d= d.reshape(d.shape[0],1)
#y3= y3.reshape(y3.shape[0],1)
#print(y3.shape)
#print(d.shape)
print("XGB accuracy: ")
print(accuracy_score(y0,c1 , normalize=True) )
#print("xg:")
#print(accuracy_score(y3, c1, normalize=True) )
