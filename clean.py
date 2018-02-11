import numpy as np 
import pandas as pd 
import csv

#df['BrandName'] = df['BrandName'].replace(['ABC', 'AB'], 'A')

df = pd.read_csv('correct.csv') #usecols = head12
df2 = pd.read_csv('rash.csv') #usecols = head12

X1 = [ 	df.TIME_OF_RECORDING,	df.TILT_ANGLE,	df.INITIAL_VELOCITY,	df.FINAL_VELOCITY, df.ACCLERATION, df.TILT_THRESHOLD,		df.SPEED_LIMIT,	df.sudden_acceleration,	df.SUDDEN_STOP,	df.EXCESS_BIKE_TILT ]


X2 =  [ 	df2.TIME_OF_RECORDING,	df2.TILT_ANGLE,	df2.INITIAL_VELOCITY,	df2.FINAL_VELOCITY,	df2.ACCLERATION,	df2.TILT_THRESHOLD,	df2.SPEED_LIMIT,	df2.sudden_acceleration,	df2.SUDDEN_STOP,	df2.EXCESS_BIKE_TILT ]
 
X = np.column_stack( X1 )
Xl = np.column_stack( X2 )

#X1 = X1.replace(['-'], '0')
#X2 = X2.replace(['-'], '0')

df2.TIME_OF_RECORDING = df2.TIME_OF_RECORDING.replace(['-'], '0')	
df2.TILT_ANGLE= df2.TILT_ANGLE.replace(['-'], '0')
df2.INITIAL_VELOCITY=df2.INITIAL_VELOCITY.replace(['-'], '0')	
df2.FINAL_VELOCITY=df2.FINAL_VELOCITY.replace(['-'], '0')
df2.ACCLERATION=df2.ACCLERATION.replace(['-'], '0')
df2.TILT_THRESHOLD= df2.TILT_THRESHOLD.replace(['-'], '0')	
df2.SPEED_LIMIT=df2.SPEED_LIMIT.replace(['-'], '0')
df2.sudden_acceleration=df2.sudden_acceleration.replace(['-'], '0')	
df2.SUDDEN_STOP=df2.SUDDEN_STOP.replace(['-'], '0')
df2.EXCESS_BIKE_TILT= df2.EXCESS_BIKE_TILT.replace(['-'], '0')
df2.to_csv("output01.csv", index=False)
