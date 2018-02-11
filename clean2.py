import serial
import matplotlib.pyplot as plt
import numpy as np
import csv


df = pd.read_csv('correct.csv') #usecols = head12
df2 = pd.read_csv('rash.csv') #usecols = head12
csvfile = open('newcorrect.csv', 'wb') #open file for operation

writer = csv.writer(csvfile) 

a=0 
while a!=120:

writer.writerow([angle, vin,vfin, accel,thresh," ",tempvin,tempaccel,tempdeaccel,tempthresh ])

a=a+1

csvfile.close()    
