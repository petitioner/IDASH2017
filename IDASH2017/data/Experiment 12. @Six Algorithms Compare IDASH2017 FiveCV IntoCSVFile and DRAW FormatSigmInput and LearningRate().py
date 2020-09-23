# coding=utf8
# 2019-12-16 15:56 a.m. GMT +08：00
'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++
+       NumPy version 1.10.2    Python 2.7.11       +
+++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import os
import math
import time
import random

from copy import deepcopy
from math import log, exp, pow, sqrt,sin,cos

import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

# Calculate the ROC-curve and the value of AUC
# INPUT: score = [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, ... ]
#        y = [1,1,0, 1, 1, 1, 0, 0, 1, 0, 1,0, 1, 0, 0, 0, 1 , 0, 1, 0]
# WARNNING: WHAT ARE THE LABELS ? {-0, +1} or {-1, +1}
def ROCAUC(score, y, show=False):
	z = zip(score,y)
	z.sort()
	score = [ x[0] for x in z ]
	y = [ x[1] for x in z ]

	# score is already in order
	thr = score

	POSITIVE = y.count(+1)
	NEGATIVE = y.count(-1)            # WARNNING: WHAT ARE THE LABELS ?
	roc_x = [1]
	roc_y = [1]
	FN = 0
	TN = 0
	# need (score,y) to be sorted
	for (i, T) in enumerate(thr):
		if y[i]==+1:
			FN = FN + 1
		if y[i]==-1:                  # WARNNING: WHAT ARE THE LABELS ?
			TN = TN + 1
		roc_x.append(1-TN/float(NEGATIVE))
		roc_y.append(1-FN/float(POSITIVE))
	z = zip(roc_x, roc_y)
	z.sort()
	roc_x = [ x[0] for x in z ]
	roc_y = [ x[1] for x in z ]
	#print zip(roc_x,roc_y)

	AUC = 0.0
	prex = roc_x[0]
	for (i, x) in enumerate(roc_x):
		AUC += (x-prex)*roc_y[i]
		prex = x
	#print AUC

	if show:
		plt.plot(roc_x, roc_y)
		plt.plot([0,1],[0,1])
		plt.axis("equal")
		plt.title('AUC = '+str(AUC))
		plt.grid(color='b' , linewidth='0.3' ,linestyle='--')
		plt.show()
	return AUC

print '----------------------------------------------------------------------------------'
print "------------- Experiment12. Six Algorithms Compare                   -------------"
print '-------------    Data Set : IDASH 2017                               -------------'
print '-------------           X : [[1,x11,x12,...],[1,x21,x22,...],...]    -------------'
print '-------------           Y : y = {-1, +1}                             -------------'
print '----------------------------------------------------------------------------------'
print "------------- Bonte's Simple Fixed Hessian Newton Method             -------------"
print "------------- Bonte's SFH with .25XTXasG and learningrate            -------------"
print "------------- Bonte's SFH with HESSIANasG and learningrate           -------------"
print "------------- Nesterov's Accelerated Gradient Descent                -------------"
print "------------- Nesterov's AG with .25XTX as Quadratic Gradient        -------------"
print "------------- Nesterov's AG with HESSIAN as Quadratic Gradient       -------------"
print '----------------------------------------------------------------------------------'

import csv

str_dataset = 'Experiment_Result_ '
#str_datetime = time.strftime("%Y-%m-%d %X", time.localtime()) 
#pathNesterov                 = '../data/testPlainResultNesterov_'
'''---------------------------- BonteSFH ----------------------------'''
resfilepath_test_BonteSFH_MLE =  str_dataset + 'TEST BonteSFH MLE.csv'
resfilepath_test_BonteSFH_AUC =  str_dataset + 'TEST BonteSFH AUC.csv'
resfilepath_train_BonteSFH_MLE = str_dataset + 'TRAIN BonteSFH MLE.csv'
resfilepath_train_BonteSFH_AUC = str_dataset + 'TRAIN BonteSFH AUC.csv'
resfilepath_BonteSFH_TIME =      str_dataset + 'BonteSFH TIME.csv'
test_BonteSFH_MLE =  open(resfilepath_test_BonteSFH_MLE,  'w')
test_BonteSFH_AUC =  open(resfilepath_test_BonteSFH_AUC,  'w')
train_BonteSFH_MLE = open(resfilepath_train_BonteSFH_MLE, 'w')
train_BonteSFH_AUC = open(resfilepath_train_BonteSFH_AUC, 'w')
BonteSFH_TIME =      open(resfilepath_BonteSFH_TIME,      'w')
test_BonteSFH_MLE =  open(resfilepath_test_BonteSFH_MLE,  'a+b')
test_BonteSFH_AUC =  open(resfilepath_test_BonteSFH_AUC,  'a+b')
train_BonteSFH_MLE = open(resfilepath_train_BonteSFH_MLE, 'a+b')
train_BonteSFH_AUC = open(resfilepath_train_BonteSFH_AUC, 'a+b')
BonteSFH_TIME =      open(resfilepath_BonteSFH_TIME,      'a+b')
'''---------------------- BonteSFH + .25XTXasG ----------------------'''
resfilepath_test_BonteSFHwithXTXasG_MLE =  str_dataset + 'TEST BonteSFHwithXTXasG MLE.csv'
resfilepath_test_BonteSFHwithXTXasG_AUC =  str_dataset + 'TEST BonteSFHwithXTXasG AUC.csv'
resfilepath_train_BonteSFHwithXTXasG_MLE = str_dataset + 'TRAIN BonteSFHwithXTXasG MLE.csv'
resfilepath_train_BonteSFHwithXTXasG_AUC = str_dataset + 'TRAIN BonteSFHwithXTXasG AUC.csv'
resfilepath_BonteSFHwithXTXasG_TIME =      str_dataset + 'BonteSFHwithXTXasG TIME.csv'
test_BonteSFHwithXTXasG_MLE =  open(resfilepath_test_BonteSFHwithXTXasG_MLE,  'w')
test_BonteSFHwithXTXasG_AUC =  open(resfilepath_test_BonteSFHwithXTXasG_AUC,  'w')
train_BonteSFHwithXTXasG_MLE = open(resfilepath_train_BonteSFHwithXTXasG_MLE, 'w')
train_BonteSFHwithXTXasG_AUC = open(resfilepath_train_BonteSFHwithXTXasG_AUC, 'w')
BonteSFHwithXTXasG_TIME =      open(resfilepath_BonteSFHwithXTXasG_TIME,      'w')
test_BonteSFHwithXTXasG_MLE =  open(resfilepath_test_BonteSFHwithXTXasG_MLE,  'a+b')
test_BonteSFHwithXTXasG_AUC =  open(resfilepath_test_BonteSFHwithXTXasG_AUC,  'a+b')
train_BonteSFHwithXTXasG_MLE = open(resfilepath_train_BonteSFHwithXTXasG_MLE, 'a+b')
train_BonteSFHwithXTXasG_AUC = open(resfilepath_train_BonteSFHwithXTXasG_AUC, 'a+b')
BonteSFHwithXTXasG_TIME =      open(resfilepath_BonteSFHwithXTXasG_TIME,      'a+b')
'''---------------------- BonteSFH + HESSIANasG ---------------------'''
resfilepath_test_BonteSFHwithHESSIAN_MLE =  str_dataset + 'TEST BonteSFHwithHESSIAN MLE.csv'
resfilepath_test_BonteSFHwithHESSIAN_AUC =  str_dataset + 'TEST BonteSFHwithHESSIAN AUC.csv'
resfilepath_train_BonteSFHwithHESSIAN_MLE = str_dataset + 'TRAIN BonteSFHwithHESSIAN MLE.csv'
resfilepath_train_BonteSFHwithHESSIAN_AUC = str_dataset + 'TRAIN BonteSFHwithHESSIAN AUC.csv'
resfilepath_BonteSFHwithHESSIAN_TIME =      str_dataset + 'BonteSFHwithHESSIAN TIME.csv'
test_BonteSFHwithHESSIAN_MLE =  open(resfilepath_test_BonteSFHwithHESSIAN_MLE,  'w')
test_BonteSFHwithHESSIAN_AUC =  open(resfilepath_test_BonteSFHwithHESSIAN_AUC,  'w')
train_BonteSFHwithHESSIAN_MLE = open(resfilepath_train_BonteSFHwithHESSIAN_MLE, 'w')
train_BonteSFHwithHESSIAN_AUC = open(resfilepath_train_BonteSFHwithHESSIAN_AUC, 'w')
BonteSFHwithHESSIAN_TIME =      open(resfilepath_BonteSFHwithHESSIAN_TIME,      'w')
test_BonteSFHwithHESSIAN_MLE =  open(resfilepath_test_BonteSFHwithHESSIAN_MLE,  'a+b')
test_BonteSFHwithHESSIAN_AUC =  open(resfilepath_test_BonteSFHwithHESSIAN_AUC,  'a+b')
train_BonteSFHwithHESSIAN_MLE = open(resfilepath_train_BonteSFHwithHESSIAN_MLE, 'a+b')
train_BonteSFHwithHESSIAN_AUC = open(resfilepath_train_BonteSFHwithHESSIAN_AUC, 'a+b')
BonteSFHwithHESSIAN_TIME =      open(resfilepath_BonteSFHwithHESSIAN_TIME,      'a+b')
'''--------------------------- NesterovAG ---------------------------'''
resfilepath_test_NesterovAG_MLE =  str_dataset + 'TEST NesterovAG MLE.csv'
resfilepath_test_NesterovAG_AUC =  str_dataset + 'TEST NesterovAG AUC.csv'
resfilepath_train_NesterovAG_MLE = str_dataset + 'TRAIN NesterovAG MLE.csv'
resfilepath_train_NesterovAG_AUC = str_dataset + 'TRAIN NesterovAG AUC.csv'
resfilepath_NesterovAG_TIME =      str_dataset + 'NesterovAG TIME.csv'
test_NesterovAG_MLE =  open(resfilepath_test_NesterovAG_MLE,  'w')
test_NesterovAG_AUC =  open(resfilepath_test_NesterovAG_AUC,  'w')
train_NesterovAG_MLE = open(resfilepath_train_NesterovAG_MLE, 'w')
train_NesterovAG_AUC = open(resfilepath_train_NesterovAG_AUC, 'w')
NesterovAG_TIME =      open(resfilepath_NesterovAG_TIME,      'w')
test_NesterovAG_MLE =  open(resfilepath_test_NesterovAG_MLE,  'a+b')
test_NesterovAG_AUC =  open(resfilepath_test_NesterovAG_AUC,  'a+b')
train_NesterovAG_MLE = open(resfilepath_train_NesterovAG_MLE, 'a+b')
train_NesterovAG_AUC = open(resfilepath_train_NesterovAG_AUC, 'a+b')
NesterovAG_TIME =      open(resfilepath_NesterovAG_TIME,      'a+b')
'''--------------------- NesterovAG + .25XTXasG ---------------------'''
resfilepath_test_NesterovAGwithXTXasG_MLE =  str_dataset + 'TEST NesterovAGwithXTXasG MLE.csv'
resfilepath_test_NesterovAGwithXTXasG_AUC =  str_dataset + 'TEST NesterovAGwithXTXasG AUC.csv'
resfilepath_train_NesterovAGwithXTXasG_MLE = str_dataset + 'TRAIN NesterovAGwithXTXasG MLE.csv'
resfilepath_train_NesterovAGwithXTXasG_AUC = str_dataset + 'TRAIN NesterovAGwithXTXasG AUC.csv'
resfilepath_NesterovAGwithXTXasG_TIME =      str_dataset + 'NesterovAGwithXTXasG TIME.csv'
test_NesterovAGwithXTXasG_MLE =  open(resfilepath_test_NesterovAGwithXTXasG_MLE,  'w')
test_NesterovAGwithXTXasG_AUC =  open(resfilepath_test_NesterovAGwithXTXasG_AUC,  'w')
train_NesterovAGwithXTXasG_MLE = open(resfilepath_train_NesterovAGwithXTXasG_MLE, 'w')
train_NesterovAGwithXTXasG_AUC = open(resfilepath_train_NesterovAGwithXTXasG_AUC, 'w')
NesterovAGwithXTXasG_TIME =      open(resfilepath_NesterovAGwithXTXasG_TIME,      'w')
test_NesterovAGwithXTXasG_MLE =  open(resfilepath_test_NesterovAGwithXTXasG_MLE,  'a+b')
test_NesterovAGwithXTXasG_AUC =  open(resfilepath_test_NesterovAGwithXTXasG_AUC,  'a+b')
train_NesterovAGwithXTXasG_MLE = open(resfilepath_train_NesterovAGwithXTXasG_MLE, 'a+b')
train_NesterovAGwithXTXasG_AUC = open(resfilepath_train_NesterovAGwithXTXasG_AUC, 'a+b')
NesterovAGwithXTXasG_TIME =      open(resfilepath_NesterovAGwithXTXasG_TIME,      'a+b')
'''---------------------- NesterovAG + HESSIAN ----------------------'''
resfilepath_test_NesterovAGwithHESSIANasG_MLE =  str_dataset + 'TEST NesterovAGwithHESSIANasG MLE.csv'
resfilepath_test_NesterovAGwithHESSIANasG_AUC =  str_dataset + 'TEST NesterovAGwithHESSIANasG AUC.csv'
resfilepath_train_NesterovAGwithHESSIANasG_MLE = str_dataset + 'TRAIN NesterovAGwithHESSIANasG MLE.csv'
resfilepath_train_NesterovAGwithHESSIANasG_AUC = str_dataset + 'TRAIN NesterovAGwithHESSIANasG AUC.csv'
resfilepath_NesterovAGwithHESSIANasG_TIME =      str_dataset + 'NesterovAGwithHESSIANasG TIME.csv'
test_NesterovAGwithHESSIANasG_MLE =  open(resfilepath_test_NesterovAGwithHESSIANasG_MLE,  'w')
test_NesterovAGwithHESSIANasG_AUC =  open(resfilepath_test_NesterovAGwithHESSIANasG_AUC,  'w')
train_NesterovAGwithHESSIANasG_MLE = open(resfilepath_train_NesterovAGwithHESSIANasG_MLE, 'w')
train_NesterovAGwithHESSIANasG_AUC = open(resfilepath_train_NesterovAGwithHESSIANasG_AUC, 'w')
NesterovAGwithHESSIANasG_TIME =      open(resfilepath_NesterovAGwithHESSIANasG_TIME,      'w')
test_NesterovAGwithHESSIANasG_MLE =  open(resfilepath_test_NesterovAGwithHESSIANasG_MLE,  'a+b')
test_NesterovAGwithHESSIANasG_AUC =  open(resfilepath_test_NesterovAGwithHESSIANasG_AUC,  'a+b')
train_NesterovAGwithHESSIANasG_MLE = open(resfilepath_train_NesterovAGwithHESSIANasG_MLE, 'a+b')
train_NesterovAGwithHESSIANasG_AUC = open(resfilepath_train_NesterovAGwithHESSIANasG_AUC, 'a+b')
NesterovAGwithHESSIANasG_TIME =      open(resfilepath_NesterovAGwithHESSIANasG_TIME,      'a+b')

# Stage 1. 
#     Step 1. Extract data from a csv file
#with open('edin.txt','r') as csvfile:
#with open('lbw.txt','r') as csvfile:
#with open('nhanes3.txt','r') as csvfile:
#with open('pcs.txt','r') as csvfile:
#with open('uis.txt','r') as csvfile:
with open('Credit_train.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	reader.next() # leave behind the first row
	data = []
	for row in reader:
		# reader.next() return a string
		row = [float(x) for x in row]
		data.append(row)
csvfile.close()
#     Step 2. Extract X and Y from data
'''           get X and Y as follows:
X = | 1 X11 X12 X13 ... X1d|
    | 1 X21 X22 X23 ... X2d|
    | .  .   .   .  ...  . |
    | .  .   .   .  ...  . |
    | 1 Xn1 Xn2 Xn3 ... Xnd|
Y = [ Y1 Y2  Y3  Y4 ... Yn ]
'''
dataset = [[1]+row[1:] for row in data[:]]
for colidx in range(len(dataset[0])):
	colmax = 1.0
	for (rowidx, row) in enumerate(dataset):
		if row[colidx] > colmax :
			colmax = row[colidx]
	for (rowidx, row) in enumerate(dataset):
		row[colidx] /= colmax
datalabel = [int(row[0]) for row in data[:]]
# turn y{+0,+1} to y{-1,+1}
datalabel = [2*y-1 for y in datalabel]    # DONT FORGET THAT THE IDASH DATASET IS DIFFERENT FROM THE MNIST DATASET!

#should shuffle [Y,X] together!
Z = zip(datalabel,dataset)
random.shuffle(Z)
dataset = [item[1] for item in Z]
datalabel = [item[0] for item in Z]

# used for the input range of sigmoid
def testx(inputx):
	#return inputx
	if inputx > 200 : 
		return 200
	if inputx < -200 :
		return -200;
	return inputx
'''
Learning rate is a decreasing function of time. 
#Two forms that are commonly used are a linear function of time 
and a function that is inversely proportional to the time t.
'''
# the learning rate for Gradient
def learningr(iter, MAX_ITER):
	#small dataset could use this learning rate
	# probably only HESSIANasG need this small learning rate
	#return 4/(1 + iter + MAX_ITER) + 1.01

	#large dataset could use this learning rate
	sigmoidx = 20*(iter+1 -1)/(MAX_ITER -1) -10
	learningrate = 1.0 + 1.0/(exp(sigmoidx) + 1.0)
	return learningrate


hlambda = lambda x:1.0/(1+exp(-x))
#hlambda = lambda x:5.0000e-01  +1.7786e-01*x  -3.6943e-03*pow(x,3)  +3.6602e-05*pow(x,5)  -1.2344e-07*pow(x,7)
MAX_ITER = 30

NUMcv = 5
NUMtest = len(dataset)/NUMcv

for idxcv in range(NUMcv):
	str_idxcv = 'CV-'+str(idxcv)+'TH: '
	'''---------------------------- BonteSFH ----------------------------'''
	test_BonteSFH_MLE.write(str_idxcv);  test_BonteSFH_MLE.write(',')
	test_BonteSFH_AUC.write(str_idxcv);  test_BonteSFH_AUC.write(','); 
	train_BonteSFH_MLE.write(str_idxcv); train_BonteSFH_MLE.write(',');
	train_BonteSFH_AUC.write(str_idxcv); train_BonteSFH_AUC.write(','); 
	BonteSFH_TIME.write(str_idxcv);      BonteSFH_TIME.write(',');
	'''---------------------- BonteSFH + .25XTXasG ----------------------'''
	test_BonteSFHwithXTXasG_MLE.write(str_idxcv);  test_BonteSFHwithXTXasG_MLE.write(',');
	test_BonteSFHwithXTXasG_AUC.write(str_idxcv);  test_BonteSFHwithXTXasG_AUC.write(',');
	train_BonteSFHwithXTXasG_MLE.write(str_idxcv); train_BonteSFHwithXTXasG_MLE.write(',');
	train_BonteSFHwithXTXasG_AUC.write(str_idxcv); train_BonteSFHwithXTXasG_AUC.write(',');
	BonteSFHwithXTXasG_TIME.write(str_idxcv);      BonteSFHwithXTXasG_TIME.write(','); 
	'''---------------------- BonteSFH + HESSIANasG ---------------------'''
	test_BonteSFHwithHESSIAN_MLE.write(str_idxcv);  test_BonteSFHwithHESSIAN_MLE.write(',');
	test_BonteSFHwithHESSIAN_AUC.write(str_idxcv);  test_BonteSFHwithHESSIAN_AUC.write(','); 
	train_BonteSFHwithHESSIAN_MLE.write(str_idxcv); train_BonteSFHwithHESSIAN_MLE.write(','); 
	train_BonteSFHwithHESSIAN_AUC.write(str_idxcv); train_BonteSFHwithHESSIAN_AUC.write(',');
	BonteSFHwithHESSIAN_TIME.write(str_idxcv);      BonteSFHwithHESSIAN_TIME.write(','); 
	'''--------------------------- NesterovAG ---------------------------'''
	test_NesterovAG_MLE.write(str_idxcv);  test_NesterovAG_MLE.write(',');
	test_NesterovAG_AUC.write(str_idxcv);  test_NesterovAG_AUC.write(',');
	train_NesterovAG_MLE.write(str_idxcv); train_NesterovAG_MLE.write(','); 
	train_NesterovAG_AUC.write(str_idxcv); train_NesterovAG_AUC.write(','); 
	NesterovAG_TIME.write(str_idxcv);      NesterovAG_TIME.write(','); 
	'''--------------------- NesterovAG + .25XTXasG ---------------------'''
	test_NesterovAGwithXTXasG_MLE.write(str_idxcv);  test_NesterovAGwithXTXasG_MLE.write(','); 
	test_NesterovAGwithXTXasG_AUC.write(str_idxcv);  test_NesterovAGwithXTXasG_AUC.write(','); 
	train_NesterovAGwithXTXasG_MLE.write(str_idxcv); train_NesterovAGwithXTXasG_MLE.write(',');
	train_NesterovAGwithXTXasG_AUC.write(str_idxcv); train_NesterovAGwithXTXasG_AUC.write(',');
	NesterovAGwithXTXasG_TIME.write(str_idxcv);      NesterovAGwithXTXasG_TIME.write(','); 
	'''---------------------- NesterovAG + HESSIAN ----------------------'''
	test_NesterovAGwithHESSIANasG_MLE.write(str_idxcv);  test_NesterovAGwithHESSIANasG_MLE.write(','); 
	test_NesterovAGwithHESSIANasG_AUC.write(str_idxcv);  test_NesterovAGwithHESSIANasG_AUC.write(',');
	train_NesterovAGwithHESSIANasG_MLE.write(str_idxcv); train_NesterovAGwithHESSIANasG_MLE.write(',');
	train_NesterovAGwithHESSIANasG_AUC.write(str_idxcv); train_NesterovAGwithHESSIANasG_AUC.write(',');
	NesterovAGwithHESSIANasG_TIME.write(str_idxcv);      NesterovAGwithHESSIANasG_TIME.write(',');
	

	X = []
	Y = []
	testX = []
	testY = []
	testX = dataset[idxcv*NUMtest:idxcv*NUMtest+NUMtest]
	testY = datalabel[idxcv*NUMtest:idxcv*NUMtest+NUMtest]
	X = dataset[0:idxcv*NUMtest] + dataset[idxcv*NUMtest+NUMtest:]
	Y = datalabel[0:idxcv*NUMtest] + datalabel[idxcv*NUMtest+NUMtest:]


	# Calculate the 2/(.25*SUM{x[i][j]}) * .9
	sumxij = 0.0
	for row in X:
		for rowi in row:
			sumxij += rowi
	sumxij = sumxij/4.0
	x0 = 2.0 / sumxij  *    .9
	print 'x0 = 2.0 / sumxij * .9 = ', x0


	'''
	-------------------------------------------------------------------------------------------
	------------------------- The Presented Method: Bonte's Simple Fixed Hessian --------------
	-------------------------------------------------------------------------------------------
	------------------------------- Bonte's Simple Fixed Hessian ------------------------------
	-------------------------------------------------------------------------------------------
	''' 
	Tbegin = time.clock()
	# Stage 2. 
	#     Step 1. Initialize Simplified Fixed Hessian Matrix
	MX = np.matrix(X)
	MXT = MX.T
	MXTMX = MXT.dot(MX)                 

	# H = +1/4 * X.T * X         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	# ADAGRAD : ϵ is a smoothing term that avoids division by zero (usually on the order of 1e−8)
	# 'cause MB already has (-.25), in this case, don't use the Quadratic Gradient Descent Method 
	epsilon = 1e-08
	# BEGIN: Bonte's Specific Order On XTX
	'''
	X = | X11 X12 X13 |     
	    | X21 X22 X23 |               
	    | X31 X32 X33 |              
	the sum of each row of (X.T * X) is a column vector as follows:
	| X11 X21 X31 |   | X11+X12+X13 | 
	| X12 X22 X32 | * | X21+X22+X23 |
	| X13 X23 X33 |   | X31+X32+X33 |
	'''
	# return a column vector whose each element is the sum of each row of MX 
	mx = MX.sum(axis=1)
	# return a column vector whose each element is the sum of each row of (X.T * X)
	print mx
	mxtmx = MX.T.dot(mx)
	print mxtmx
	mb = np.matrix(np.eye(mxtmx.shape[0]))
	for idx in range(mxtmx.shape[0]):
		mb[idx,idx] = (+.25)*mxtmx[idx, 0]  + (+.25)*epsilon           
		print 'M[',idx,'][',idx,'] = ',mb[idx,idx]
	# END  : Bonte's Specific Order On XTX

	MB = mb    
	print MB 
	# Use Newton Method to calculate the inverses of MB[i][i] 
	MB_inv = np.matrix(np.eye(mxtmx.shape[0]))
	for idx in range(mxtmx.shape[0]):
		MB_inv[idx,idx] = 1.0/MB[idx,idx]


	#     Step 2. Initialize Weight Vector (n x 1)
	# Setting the initial weight to 1 leads to a large input to sigmoid function,
	# which would cause a big problem to this algorithm when using polynomial
	# to substitute the sigmoid function. So, it is a good choice to set w = 0.

	# [[0]... to make MW a column vector(matrix)
	W = [[0] for x in range(MB.shape[0])]
	MW = np.matrix(W)

	Tend = time.clock()
	#     Step 2. Set the Maximum Iteration and Record each cost function
	Emethod_TEST_Bonte_MLE = []
	Emethod_TEST_Bonte_AUC = []
	Emethod_TRAIN_Bonte_MLE = []
	Emethod_TRAIN_Bonte_AUC = []
	EmethodBonte_SIGMOID = []
	EmethodBonte_TIME = []

	EmethodBonte_TIME.append(Tend - Tbegin)
	BonteSFH_TIME.write(str(Tend - Tbegin));      BonteSFH_TIME.write(',');
	# Calculate test dataset initial log MLE
	# log-likelihood
	MtestX = np.matrix(testX)
	newMtestXV = MtestX * MW
	loghx = []
	for idx in range(len(testY)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TEST_Bonte_MLE.append(loglikelihood)
	test_BonteSFH_MLE.write(str(loglikelihood)); test_BonteSFH_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(testY)):
		hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TEST_Bonte_AUC.append(ROCAUC(hxlist, testY))
	test_BonteSFH_AUC.write(str(ROCAUC(hxlist, testY))); test_BonteSFH_AUC.write(",")
	#---------------------------------------------------------------------#
	# log-likelihood
	MX = np.matrix(X)
	newMXW = MX * MW
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TRAIN_Bonte_MLE.append(loglikelihood)
	train_BonteSFH_MLE.write(str(loglikelihood)); train_BonteSFH_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TRAIN_Bonte_AUC.append(ROCAUC(hxlist, Y))
	train_BonteSFH_AUC.write(str(ROCAUC(hxlist, Y))); train_BonteSFH_AUC.write(",")


	# Stage 3.
	#     Start the Gradient Descent algorithm
	#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	#           grad = [Y@(1 - sigm(yWTx))]T * X
	for iter in range(MAX_ITER):
		Tbegin = time.clock()
		curSigmoidInput = []

	#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
		# W.T * X
		MXW = MX * MW
		# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
		yhypothesis = []
		for idx in range(len(Y)):

			curSigmoidInput.append(Y[idx]*MXW.A[idx][0])

			# hlambda(): the polynomial function to substitute the Sigmoid function
			h = 1 - hlambda(testx(Y[idx]*MXW.A[idx][0]))

			yhypothesis.append([h*Y[idx]])

		Myhypothesis = np.matrix(yhypothesis)
		# g = [Y@(1 - sigm(yWTx))]T * X	
		Mg = MXT * Myhypothesis
		
		MG = MB_inv * Mg          
		# should be 'plus', 'cause to compute the MLE
		MW = MW + MG       

		Tend = time.clock()
		#Ebonte_TIME.append(Tend - Tbegin + Ebonte_TIME[-1])
		EmethodBonte_TIME.append(Tend - Tbegin)
		BonteSFH_TIME.write(str(Tend - Tbegin))
		if iter+1!=MAX_ITER : BonteSFH_TIME.write(",")


		EmethodBonte_SIGMOID.append(curSigmoidInput)         
	#     Step 4. Calculate the cost function using Maximum likelihood Estimation
		# log-likelihood
		MtestX = np.matrix(testX)
		newMtestXV = MtestX * MW
		loghx = []
		for idx in range(len(testY)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TEST_Bonte_MLE.append(loglikelihood)
		test_BonteSFH_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : test_BonteSFH_MLE.write(",")


		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(testY)):
			hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TEST_Bonte_AUC.append(ROCAUC(hxlist, testY))
		test_BonteSFH_AUC.write(str(ROCAUC(hxlist, testY)))
		if iter+1!=MAX_ITER : test_BonteSFH_AUC.write(",")
		#print iter, '-th AUC : ', ROCAUC(hxlist, testY), ' MLE = ', loglikelihood
		

		#---------------------------------------------------------------------#
		# log-likelihood
		MX = np.matrix(X)
		newMXW = MX * MW
		loghx = []
		for idx in range(len(Y)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TRAIN_Bonte_MLE.append(loglikelihood)
		train_BonteSFH_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : train_BonteSFH_MLE.write(",")
		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(Y)):
			hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TRAIN_Bonte_AUC.append(ROCAUC(hxlist, Y))
		train_BonteSFH_AUC.write(str(ROCAUC(hxlist, Y)))
		if iter+1!=MAX_ITER : train_BonteSFH_AUC.write(",")
		print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood

	'''
	-------------------------------------------------------------------------------------------
	----------------------------------- end of experiments -----------------------------------
	-------------------------------------------------------------------------------------------
	'''

	'''
	-------------------------------------------------------------------------------------------
	------------------------- The Presented Method: Bonte + .25XTXasG -------------------------
	-------------------------------------------------------------------------------------------
	--------------- Bonte's Simple Fixed Hessian with G (SFH directly by .25XTX) --------------
	-------------------------------------------------------------------------------------------
	''' 
	Tbegin = time.clock()
	# Stage 2. 
	#     Step 1. Initialize Simplified Fixed Hessian Matrix
	MX = np.matrix(X)
	MXT = MX.T
	MXTMX = MXT.dot(MX)                 

	# H = +1/4 * X.T * X         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	# ADAGRAD : ϵ is a smoothing term that avoids division by zero (usually on the order of 1e−8)
	# 'cause MB already has (-.25), in this case, don't use the Quadratic Gradient Descent Method 
	epsilon = 1e-08
	# BEGIN: Bonte's Specific Order On XTX
	'''
	X = | X11 X12 X13 |     
	    | X21 X22 X23 |               
	    | X31 X32 X33 |              
	the sum of each row of (X.T * X) is a column vector as follows:
	| X11 X21 X31 |   | X11+X12+X13 | 
	| X12 X22 X32 | * | X21+X22+X23 |
	| X13 X23 X33 |   | X31+X32+X33 |
	'''
	# return a column vector whose each element is the sum of each row of MX 
	mx = MX.sum(axis=1)
	# return a column vector whose each element is the sum of each row of (X.T * X)
	print mx
	mxtmx = MX.T.dot(mx)
	print mxtmx
	mb = np.matrix(np.eye(mxtmx.shape[0]))
	for idx in range(mxtmx.shape[0]):
		mb[idx,idx] = (+.25)*mxtmx[idx, 0]  + (+.25)*epsilon           
		print 'M[',idx,'][',idx,'] = ',mb[idx,idx]
	# END  : Bonte's Specific Order On XTX

	MB = mb    
	print MB 
	# Use Newton Method to calculate the inverses of MB[i][i] 
	MB_inv = np.matrix(np.eye(mxtmx.shape[0]))
	for idx in range(mxtmx.shape[0]):
		MB_inv[idx,idx] = 1.0/MB[idx,idx]


	#     Step 2. Initialize Weight Vector (n x 1)
	# Setting the initial weight to 1 leads to a large input to sigmoid function,
	# which would cause a big problem to this algorithm when using polynomial
	# to substitute the sigmoid function. So, it is a good choice to set w = 0.

	# [[0]... to make MW a column vector(matrix)
	W = [[0] for x in range(MB.shape[0])]
	MW = np.matrix(W)

	Tend = time.clock()
	#     Step 2. Set the Maximum Iteration and Record each cost function
	Emethod_TEST_BonteWithXTXasG_MLE = []
	Emethod_TEST_BonteWithXTXasG_AUC = []
	Emethod_TRAIN_BonteWithXTXasG_MLE = []
	Emethod_TRAIN_BonteWithXTXasG_AUC = []
	EmethodBonteWithXTXasG_SIGMOID = []
	EmethodBonteWithXTXasG_TIME = []

	EmethodBonteWithXTXasG_TIME.append(Tend - Tbegin)
	BonteSFHwithXTXasG_TIME.write(str(Tend - Tbegin)); BonteSFHwithXTXasG_TIME.write(",")
	# log-likelihood
	MtestX = np.matrix(testX)
	newMtestXV = MtestX * MW
	loghx = []
	for idx in range(len(testY)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TEST_BonteWithXTXasG_MLE.append(loglikelihood)
	test_BonteSFHwithXTXasG_MLE.write(str(loglikelihood)); test_BonteSFHwithXTXasG_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(testY)):
		hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TEST_BonteWithXTXasG_AUC.append(ROCAUC(hxlist, testY))
	test_BonteSFHwithXTXasG_AUC.write(str(ROCAUC(hxlist, testY))); test_BonteSFHwithXTXasG_AUC.write(",")
	#---------------------------------------------------------------------#
	# log-likelihood
	MX = np.matrix(X)
	newMXW = MX * MW
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TRAIN_BonteWithXTXasG_MLE.append(loglikelihood)
	train_BonteSFHwithXTXasG_MLE.write(str(loglikelihood)); train_BonteSFHwithXTXasG_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TRAIN_BonteWithXTXasG_AUC.append(ROCAUC(hxlist, Y))
	train_BonteSFHwithXTXasG_AUC.write(str(ROCAUC(hxlist, Y))); train_BonteSFHwithXTXasG_AUC.write(",")
	
	# Stage 3.
	#     Start the Gradient Descent algorithm
	#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	#           grad = [Y@(1 - sigm(yWTx))]T * X
	for iter in range(MAX_ITER):
		Tbegin = time.clock()
		curSigmoidInput = []

	#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
		# W.T * X
		MXW = MX * MW
		# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
		yhypothesis = []
		for idx in range(len(Y)):

			curSigmoidInput.append(Y[idx]*MXW.A[idx][0])

			# hlambda(): the polynomial function to substitute the Sigmoid function
			h = 1 - hlambda(testx(Y[idx]*MXW.A[idx][0]))

			yhypothesis.append([h*Y[idx]])

		Myhypothesis = np.matrix(yhypothesis)
		# g = [Y@(1 - sigm(yWTx))]T * X	
		Mg = MXT * Myhypothesis
		
		MG = MB_inv * Mg          
		# should be 'plus', 'cause to compute the MLE
		# rescale Iter into [-10,+10] using x = (10--10)*(iter-1)/(MaxIter-1) + -10
		# the learning rate = 2 - y = 2 - 1./(1+e^(-x)) = 1 + 1/(e^x + 1)
		#sigmoidx = 20*(iter+1 -1)/(MAX_ITER -1) -10
		#learningrate = 1.0 + 1.0/(exp(sigmoidx) + 1.0)
		#learningrate = 2
		learningrate = learningr(iter, MAX_ITER)
		MW = MW + learningrate*MG       

		Tend = time.clock()
		#Ebonte_TIME.append(Tend - Tbegin + Ebonte_TIME[-1])
		EmethodBonteWithXTXasG_TIME.append(Tend - Tbegin)
		BonteSFHwithXTXasG_TIME.write(str(Tend - Tbegin))
		if iter+1!=MAX_ITER : BonteSFHwithXTXasG_TIME.write(",")

		EmethodBonteWithXTXasG_SIGMOID.append(curSigmoidInput)         
	#     Step 4. Calculate the cost function using Maximum likelihood Estimation
		# log-likelihood
		MtestX = np.matrix(testX)
		newMtestXV = MtestX * MW
		loghx = []
		for idx in range(len(testY)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TEST_BonteWithXTXasG_MLE.append(loglikelihood)
		test_BonteSFHwithXTXasG_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : test_BonteSFHwithXTXasG_MLE.write(",")

		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(testY)):
			hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TEST_BonteWithXTXasG_AUC.append(ROCAUC(hxlist, testY))
		test_BonteSFHwithXTXasG_AUC.write(str(ROCAUC(hxlist, testY)))
		if iter+1!=MAX_ITER : test_BonteSFHwithXTXasG_AUC.write(",")
		#print iter, '-th AUC : ', ROCAUC(hxlist, testY), ' MLE = ', loglikelihood
		

		#---------------------------------------------------------------------#
		# log-likelihood
		MX = np.matrix(X)
		newMXW = MX * MW
		loghx = []
		for idx in range(len(Y)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TRAIN_BonteWithXTXasG_MLE.append(loglikelihood)
		train_BonteSFHwithXTXasG_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : train_BonteSFHwithXTXasG_MLE.write(",")
		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(Y)):
			hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TRAIN_BonteWithXTXasG_AUC.append(ROCAUC(hxlist, Y))
		train_BonteSFHwithXTXasG_AUC.write(str(ROCAUC(hxlist, Y)))
		if iter+1!=MAX_ITER : train_BonteSFHwithXTXasG_AUC.write(",")
		print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	'''
	-------------------------------------------------------------------------------------------
	----------------------------------- end of experiments -----------------------------------
	-------------------------------------------------------------------------------------------
	'''


	'''
	-------------------------------------------------------------------------------------------
	------------------------- The Presented Method: Bonte + HESSIANasG ------------------------
	-------------------------------------------------------------------------------------------
	--------------- Bonte's Simple Fixed Hessian with G (SFH directly by HESSIAN) -------------
	-------------------------------------------------------------------------------------------
	''' 
	Tbegin = time.clock()
	# Stage 2. 
	#     Step 1. Initialize Simplified Fixed Hessian Matrix
	MX = np.matrix(X)
	              

	# H = +1/4 * X.T * X         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	# ADAGRAD : ϵ is a smoothing term that avoids division by zero (usually on the order of 1e−8)
	# 'cause MB already has (-.25), in this case, don't use the Quadratic Gradient Descent Method 
	epsilon = 1e-08

	#     Step 2. Initialize Weight Vector (n x 1)
	# Setting the initial weight to 1 leads to a large input to sigmoid function,
	# which would cause a big problem to this algorithm when using polynomial
	# to substitute the sigmoid function. So, it is a good choice to set w = 0.

	# [[0]... to make MW a column vector(matrix)
	W = [[0] for x in range(MB.shape[0])]
	MW = np.matrix(W)

	Tend = time.clock()
	#     Step 2. Set the Maximum Iteration and Record each cost function
	Emethod_TEST_BonteWithHESSIANasG_MLE = []
	Emethod_TEST_BonteWithHESSIANasG_AUC = []
	Emethod_TRAIN_BonteWithHESSIANasG_MLE = []
	Emethod_TRAIN_BonteWithHESSIANasG_AUC = []
	EmethodBonteWithHESSIANasG_SIGMOID = []
	EmethodBonteWithHESSIANasG_TIME = []

	EmethodBonteWithHESSIANasG_TIME.append(Tend - Tbegin)
	BonteSFHwithHESSIAN_TIME.write(str(Tend - Tbegin)); BonteSFHwithHESSIAN_TIME.write(",")
	# log-likelihood
	MtestX = np.matrix(testX)
	newMtestXV = MtestX * MW
	loghx = []
	for idx in range(len(testY)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TEST_BonteWithHESSIANasG_MLE.append(loglikelihood)
	test_BonteSFHwithHESSIAN_MLE.write(str(loglikelihood)); test_BonteSFHwithHESSIAN_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(testY)):
		hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TEST_BonteWithHESSIANasG_AUC.append(ROCAUC(hxlist, testY))
	test_BonteSFHwithHESSIAN_AUC.write(str(ROCAUC(hxlist, testY))); test_BonteSFHwithHESSIAN_AUC.write(",")
	#---------------------------------------------------------------------#
	# log-likelihood
	MX = np.matrix(X)
	newMXW = MX * MW
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TRAIN_BonteWithHESSIANasG_MLE.append(loglikelihood)
	train_BonteSFHwithHESSIAN_MLE.write(str(loglikelihood)); train_BonteSFHwithHESSIAN_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TRAIN_BonteWithHESSIANasG_AUC.append(ROCAUC(hxlist, Y))
	train_BonteSFHwithHESSIAN_AUC.write(str(ROCAUC(hxlist, Y))); train_BonteSFHwithHESSIAN_AUC.write(",")
	
	# Stage 3.
	#     Start the Gradient Descent algorithm
	#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	#           grad = [Y@(1 - sigm(yWTx))]T * X
	for iter in range(MAX_ITER):
		break
		Tbegin = time.clock()
		curSigmoidInput = []

	#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
		# W.T * X
		MXW = MX * MW
		# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
		yhypothesis = []
		for idx in range(len(Y)):

			curSigmoidInput.append(Y[idx]*MXW.A[idx][0])

			# hlambda(): the polynomial function to substitute the Sigmoid function
			h = 1 - hlambda(testx(Y[idx]*MXW.A[idx][0]))

			yhypothesis.append([h*Y[idx]])

		Myhypothesis = np.matrix(yhypothesis)
		# g = [Y@(1 - sigm(yWTx))]T * X	
		Mg = MXT * Myhypothesis
		
		# Construct the Hessian matrix = - MXT * MMID * MX
		hypothesis = [ [1.0/(1+exp(-testx(wTx)))] for wTx in [x[0] for x in MXW.A] ]
		MMID = np.matrix(np.eye(MX.shape[0]))
		for idx in range(MMID.shape[0]):
			MMID[idx,idx] = hypothesis[idx][0] * (1-hypothesis[idx][0])
		MHHES = -1 * MXT * MMID * MX

		MB2 = np.matrix(np.eye(MHHES.shape[0]))
		MB2_inv = np.matrix(np.eye(MHHES.shape[0]))
		for rowidx in range(MB2.shape[0]):
			temp = 0.0
			for colidx in range(MB2.shape[1]):
				temp += abs(MHHES[rowidx,colidx])
			MB2_inv[rowidx,rowidx] = 1.0/(temp+epsilon)

		MG = MB2_inv * Mg          
		# should be 'plus', 'cause to compute the MLE
		# rescale Iter into [-10,+10] using x = (10--10)*(iter-1)/(MaxIter-1) + -10
		# the learning rate = 2 - y = 2 - 1./(1+e^(-x)) = 1 + 1/(e^x + 1)
		#sigmoidx = 20*(iter+1 -1)/(MAX_ITER -1) -10
		#learningrate = 1.0 + 1.0/(exp(sigmoidx) + 1.0)
		#learningrate = 2
		learningrate = learningr(iter, MAX_ITER)
		MW = MW + learningrate*MG   
		#MW = MW + MG     

		Tend = time.clock()
		#Ebonte_TIME.append(Tend - Tbegin + Ebonte_TIME[-1])
		EmethodBonteWithHESSIANasG_TIME.append(Tend - Tbegin)
		BonteSFHwithHESSIAN_TIME.write(str(Tend - Tbegin))
		if iter+1!=MAX_ITER : BonteSFHwithHESSIAN_TIME.write(",")

		EmethodBonteWithHESSIANasG_SIGMOID.append(curSigmoidInput)         
	#     Step 4. Calculate the cost function using Maximum likelihood Estimation
		# log-likelihood
		MtestX = np.matrix(testX)
		newMtestXV = MtestX * MW
		loghx = []
		for idx in range(len(testY)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TEST_BonteWithHESSIANasG_MLE.append(loglikelihood)
		test_BonteSFHwithHESSIAN_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : test_BonteSFHwithHESSIAN_MLE.write(",")

		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(testY)):
			hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TEST_BonteWithHESSIANasG_AUC.append(ROCAUC(hxlist, testY))
		test_BonteSFHwithHESSIAN_AUC.write(str(ROCAUC(hxlist, testY)))
		if iter+1!=MAX_ITER : test_BonteSFHwithHESSIAN_AUC.write(",")
		#print iter, '-th AUC : ', ROCAUC(hxlist, testY), ' MLE = ', loglikelihood
		

		#---------------------------------------------------------------------#
		# log-likelihood
		MX = np.matrix(X)
		newMXW = MX * MW
		loghx = []
		for idx in range(len(Y)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TRAIN_BonteWithHESSIANasG_MLE.append(loglikelihood)
		train_BonteSFHwithHESSIAN_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : train_BonteSFHwithHESSIAN_MLE.write(",")
		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(Y)):
			hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TRAIN_BonteWithHESSIANasG_AUC.append(ROCAUC(hxlist, Y))
		train_BonteSFHwithHESSIAN_AUC.write(str(ROCAUC(hxlist, Y)))
		if iter+1!=MAX_ITER : train_BonteSFHwithHESSIAN_AUC.write(",")
		print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	'''
	-------------------------------------------------------------------------------------------
	----------------------------------- end of experiments -----------------------------------
	-------------------------------------------------------------------------------------------
	'''

	'''
	-------------------------------------------------------------------------------------------
	------------------------- The Presented Method: Nesterov AG -------------------------------
	-------------------------------------------------------------------------------------------
	------------------------- Nesterov’s Accelerated Gradient Descent -------------------------
	-------------------------------------------------------------------------------------------
	''' 
	Tbegin = time.clock()
	# Stage 2. 
	#     Step 1. Initialize Simplified Fixed Hessian Matrix
	MX = np.matrix(X)               

	# H = +1/4 * X.T * X         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	# ADAGRAD : ϵ is a smoothing term that avoids division by zero (usually on the order of 1e−8)
	# 'cause MB already has (-.25), in this case, don't use the Quadratic Gradient Descent Method 
	epsilon = 1e-08

	#     Step 2. Initialize Weight Vector (n x 1)
	# Setting the initial weight to 1 leads to a large input to sigmoid function,
	# which would cause a big problem to this algorithm when using polynomial
	# to substitute the sigmoid function. So, it is a good choice to set w = 0.

	# [[0]... to make MW a column vector(matrix)
	V = [[0.0] for x in range(MB.shape[0])]
	W = [[0.0] for x in range(MB.shape[0])]
	MV = np.matrix(V)
	MW = np.matrix(W)

	Tend = time.clock()
	#     Step 2. Set the Maximum Iteration and Record each cost function
	Emethod_TEST_Nesterov_MLE = []
	Emethod_TEST_Nesterov_AUC = []
	Emethod_TRAIN_Nesterov_MLE = []
	Emethod_TRAIN_Nesterov_AUC = []
	EmethodNesterov_SIGMOID = []
	EmethodNesterov_TIME = []

	EmethodNesterov_TIME.append(Tend - Tbegin)
	NesterovAG_TIME.write(str(Tend - Tbegin)); NesterovAG_TIME.write(",")
	# log-likelihood
	MtestX = np.matrix(testX)
	newMtestXV = MtestX * MV
	loghx = []
	for idx in range(len(testY)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TEST_Nesterov_MLE.append(loglikelihood)
	test_NesterovAG_MLE.write(str(loglikelihood)); test_NesterovAG_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(testY)):
		hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TEST_Nesterov_AUC.append(ROCAUC(hxlist, testY))
	test_NesterovAG_AUC.write(str(ROCAUC(hxlist, testY))); test_NesterovAG_AUC.write(",")
	#---------------------------------------------------------------------#
	# log-likelihood
	MX = np.matrix(X)
	newMXW = MX * MV
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TRAIN_Nesterov_MLE.append(loglikelihood)
	train_NesterovAG_MLE.write(str(loglikelihood)); train_NesterovAG_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TRAIN_Nesterov_AUC.append(ROCAUC(hxlist, Y))
	train_NesterovAG_AUC.write(str(ROCAUC(hxlist, Y))); train_NesterovAG_AUC.write(",")

	alpha0 = 0.01
	alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0
	# Stage 3.
	#     Start the Gradient Descent algorithm
	#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	#           grad = [Y@(1 - sigm(yWTx))]T * X
	for iter in range(MAX_ITER):
		Tbegin = time.clock()
		curSigmoidInput = []

	#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
		# W.T * X
		MXV = MX * MV
		# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
		yhypothesis = []
		for idx in range(len(Y)):

			curSigmoidInput.append(Y[idx]*MXV.A[idx][0])

			# hlambda(): the polynomial function to substitute the Sigmoid function
			h = 1 - hlambda(testx(Y[idx]*MXV.A[idx][0]))

			yhypothesis.append([h*Y[idx]])

		Myhypothesis = np.matrix(yhypothesis)
		# g = [Y@(1 - sigm(yWTx))]T * X	
		Mg = MXT * Myhypothesis

		eta = (1 - alpha0) / alpha1
		gamma = 1.0/(iter+1)/MX.shape[0]
		
		#MG = MB_inv * Mg          
		# should be 'plus', 'cause to compute the MLE
		MtmpW = MV + (gamma + 0.0) * Mg           
		MV = (1.0-eta)*MtmpW + (eta)*MW
		MW = MtmpW

		alpha0 = alpha1
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0

		Tend = time.clock()
		#Ebonte_TIME.append(Tend - Tbegin + Ebonte_TIME[-1])
		EmethodNesterov_TIME.append(Tend - Tbegin)
		NesterovAG_TIME.write(str(Tend - Tbegin))
		if iter+1!=MAX_ITER : NesterovAG_TIME.write(",")

		EmethodNesterov_SIGMOID.append(curSigmoidInput)         
	#     Step 4. Calculate the cost function using Maximum likelihood Estimation
		# log-likelihood
		MtestX = np.matrix(testX)
		newMtestXV = MtestX * MV
		loghx = []
		for idx in range(len(testY)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TEST_Nesterov_MLE.append(loglikelihood)
		test_NesterovAG_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : test_NesterovAG_MLE.write(",")

		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(testY)):
			hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TEST_Nesterov_AUC.append(ROCAUC(hxlist, testY))
		test_NesterovAG_AUC.write(str(ROCAUC(hxlist, testY)))
		if iter+1!=MAX_ITER : test_NesterovAG_AUC.write(",")
		#print iter, '-th AUC : ', ROCAUC(hxlist, testY), ' MLE = ', loglikelihood
		

		#---------------------------------------------------------------------#
		# log-likelihood
		MX = np.matrix(X)
		newMXW = MX * MV
		loghx = []
		for idx in range(len(Y)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TRAIN_Nesterov_MLE.append(loglikelihood)
		train_NesterovAG_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : train_NesterovAG_MLE.write(",")
		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(Y)):
			hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TRAIN_Nesterov_AUC.append(ROCAUC(hxlist, Y))
		train_NesterovAG_AUC.write(str(ROCAUC(hxlist, Y)))
		if iter+1!=MAX_ITER : train_NesterovAG_AUC.write(",")
		print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	'''
	-------------------------------------------------------------------------------------------
	----------------------------------- end of experiments -----------------------------------
	-------------------------------------------------------------------------------------------
	'''

	'''
	-------------------------------------------------------------------------------------------
	------------------------- The Presented Method: Nesterov + .25XTXasG ----------------------
	-------------------------------------------------------------------------------------------
	------------------------ Nesterov with G (SFH directly by 0.25XTX) ------------------------
	-------------------------------------------------------------------------------------------
	''' 
	Tbegin = time.clock()
	# Stage 2. 
	#     Step 1. Initialize Simplified Fixed Hessian Matrix
	MX = np.matrix(X)
	MXT = MX.T
	MXTMX = MXT.dot(MX)                 

	# H = +1/4 * X.T * X         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	# ADAGRAD : ϵ is a smoothing term that avoids division by zero (usually on the order of 1e−8)
	# 'cause MB already has (-.25), in this case, don't use the Quadratic Gradient Descent Method 
	epsilon = 1e-08
	# BEGIN: Bonte's Specific Order On XTX
	'''
	X = | X11 X12 X13 |     
	    | X21 X22 X23 |               
	    | X31 X32 X33 |              
	the sum of each row of (X.T * X) is a column vector as follows:
	| X11 X21 X31 |   | X11+X12+X13 | 
	| X12 X22 X32 | * | X21+X22+X23 |
	| X13 X23 X33 |   | X31+X32+X33 |
	'''
	# return a column vector whose each element is the sum of each row of MX 
	mx = MX.sum(axis=1)
	# return a column vector whose each element is the sum of each row of (X.T * X)
	print mx
	mxtmx = MX.T.dot(mx)
	print mxtmx
	mb = np.matrix(np.eye(mxtmx.shape[0]))
	for idx in range(mxtmx.shape[0]):
		mb[idx,idx] = (+.25)*mxtmx[idx, 0]  + (+.25)*epsilon           
		print 'M[',idx,'][',idx,'] = ',mb[idx,idx]
	# END  : Bonte's Specific Order On XTX

	MB = mb    
	print MB 
	# Use Newton Method to calculate the inverses of MB[i][i] 
	MB_inv = np.matrix(np.eye(mxtmx.shape[0]))
	for idx in range(mxtmx.shape[0]):
		MB_inv[idx,idx] = 1.0/MB[idx,idx]


	#     Step 2. Initialize Weight Vector (n x 1)
	# Setting the initial weight to 1 leads to a large input to sigmoid function,
	# which would cause a big problem to this algorithm when using polynomial
	# to substitute the sigmoid function. So, it is a good choice to set w = 0.

	# [[0]... to make MW a column vector(matrix)
	V = [[0.0] for x in range(MB.shape[0])]
	W = [[0.0] for x in range(MB.shape[0])]
	MV = np.matrix(V)
	MW = np.matrix(W)

	Tend = time.clock()
	#     Step 2. Set the Maximum Iteration and Record each cost function
	Emethod_TEST_NesterovWithXTXasG_MLE = []
	Emethod_TEST_NesterovWithXTXasG_AUC = []
	Emethod_TRAIN_NesterovWithXTXasG_MLE = []
	Emethod_TRAIN_NesterovWithXTXasG_AUC = []
	EmethodNesterovWithXTXasG_SIGMOID = []
	EmethodNesterovWithXTXasG_TIME = []

	EmethodNesterovWithXTXasG_TIME.append(Tend - Tbegin)
	NesterovAGwithXTXasG_TIME.write(str(Tend - Tbegin)); NesterovAGwithXTXasG_TIME.write(",")
	# log-likelihood
	MtestX = np.matrix(testX)
	newMtestXV = MtestX * MV
	loghx = []
	for idx in range(len(testY)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TEST_NesterovWithXTXasG_MLE.append(loglikelihood)
	test_NesterovAGwithXTXasG_MLE.write(str(loglikelihood)); test_NesterovAGwithXTXasG_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(testY)):
		hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TEST_NesterovWithXTXasG_AUC.append(ROCAUC(hxlist, testY))
	test_NesterovAGwithXTXasG_AUC.write(str(ROCAUC(hxlist, testY))); test_NesterovAGwithXTXasG_AUC.write(",")
	#---------------------------------------------------------------------#
	# log-likelihood
	MX = np.matrix(X)
	newMXW = MX * MV
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TRAIN_NesterovWithXTXasG_MLE.append(loglikelihood)
	train_NesterovAGwithXTXasG_MLE.write(str(loglikelihood)); train_NesterovAGwithXTXasG_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TRAIN_NesterovWithXTXasG_AUC.append(ROCAUC(hxlist, Y))
	train_NesterovAGwithXTXasG_AUC.write(str(ROCAUC(hxlist, Y))); train_NesterovAGwithXTXasG_AUC.write(",")

	alpha0 = 0.01
	alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0
	# Stage 3.
	#     Start the Gradient Descent algorithm
	#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	#           grad = [Y@(1 - sigm(yWTx))]T * X
	for iter in range(MAX_ITER):
		Tbegin = time.clock()
		curSigmoidInput = []

	#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
		# W.T * X
		MXV = MX * MV
		# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
		yhypothesis = []
		for idx in range(len(Y)):

			curSigmoidInput.append(Y[idx]*MXV.A[idx][0])

			# hlambda(): the polynomial function to substitute the Sigmoid function
			h = 1 - hlambda(testx(Y[idx]*MXV.A[idx][0]))

			yhypothesis.append([h*Y[idx]])

		Myhypothesis = np.matrix(yhypothesis)
		# g = [Y@(1 - sigm(yWTx))]T * X	
		Mg = MXT * Myhypothesis

		eta = (1 - alpha0) / alpha1
		gamma = 1.0/(iter+1)/MX.shape[0]
		
		MG = MB_inv * Mg          
		# should be 'plus', 'cause to compute the MLE
		MtmpW = MV + (gamma + 1.0) * MG           
		MV = (1.0-eta)*MtmpW + (eta)*MW
		MW = MtmpW

		alpha0 = alpha1
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0

		Tend = time.clock()
		#Ebonte_TIME.append(Tend - Tbegin + Ebonte_TIME[-1])
		EmethodNesterovWithXTXasG_TIME.append(Tend - Tbegin)
		NesterovAGwithXTXasG_TIME.write(str(Tend - Tbegin))
		if iter+1!=MAX_ITER : NesterovAGwithXTXasG_TIME.write(",")

		EmethodNesterovWithXTXasG_SIGMOID.append(curSigmoidInput)         
	#     Step 4. Calculate the cost function using Maximum likelihood Estimation
		# log-likelihood
		MtestX = np.matrix(testX)
		newMtestXV = MtestX * MV
		loghx = []
		for idx in range(len(testY)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TEST_NesterovWithXTXasG_MLE.append(loglikelihood)
		test_NesterovAGwithXTXasG_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : test_NesterovAGwithXTXasG_MLE.write(",")

		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(testY)):
			hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TEST_NesterovWithXTXasG_AUC.append(ROCAUC(hxlist, testY))
		test_NesterovAGwithXTXasG_AUC.write(str(ROCAUC(hxlist, testY)))
		if iter+1!=MAX_ITER : test_NesterovAGwithXTXasG_AUC.write(",")
		#print iter, '-th AUC : ', ROCAUC(hxlist, testY), ' MLE = ', loglikelihood
		

		#---------------------------------------------------------------------#
		# log-likelihood
		MX = np.matrix(X)
		newMXW = MX * MV
		loghx = []
		for idx in range(len(Y)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TRAIN_NesterovWithXTXasG_MLE.append(loglikelihood)
		train_NesterovAGwithXTXasG_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : train_NesterovAGwithXTXasG_MLE.write(",")
		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(Y)):
			hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TRAIN_NesterovWithXTXasG_AUC.append(ROCAUC(hxlist, Y))
		train_NesterovAGwithXTXasG_AUC.write(str(ROCAUC(hxlist, Y)))
		if iter+1!=MAX_ITER : train_NesterovAGwithXTXasG_AUC.write(",")
		print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	'''
	-------------------------------------------------------------------------------------------
	----------------------------------- end of experiments -----------------------------------
	-------------------------------------------------------------------------------------------
	'''

	'''
	-------------------------------------------------------------------------------------------
	------------------------- The Presented Method: Nesterov + HESSIANasG ---------------------
	-------------------------------------------------------------------------------------------
	------------------------ Nesterov with G (SFH directly by HESSIAN) ------------------------
	-------------------------------------------------------------------------------------------
	''' 
	Tbegin = time.clock()
	# Stage 2. 
	#     Step 1. Initialize Simplified Fixed Hessian Matrix
	MX = np.matrix(X)               

	# H = +1/4 * X.T * X         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	# ADAGRAD : ϵ is a smoothing term that avoids division by zero (usually on the order of 1e−8)
	# 'cause MB already has (-.25), in this case, don't use the Quadratic Gradient Descent Method 
	epsilon = 1e-08

	#     Step 2. Initialize Weight Vector (n x 1)
	# Setting the initial weight to 1 leads to a large input to sigmoid function,
	# which would cause a big problem to this algorithm when using polynomial
	# to substitute the sigmoid function. So, it is a good choice to set w = 0.

	# [[0]... to make MW a column vector(matrix)
	V = [[0.0] for x in range(MB.shape[0])]
	W = [[0.0] for x in range(MB.shape[0])]
	MV = np.matrix(V)
	MW = np.matrix(W)

	Tend = time.clock()
	#     Step 2. Set the Maximum Iteration and Record each cost function
	Emethod_TEST_NesterovWithHESSIANasG_MLE = []
	Emethod_TEST_NesterovWithHESSIANasG_AUC = []
	Emethod_TRAIN_NesterovWithHESSIANasG_MLE = []
	Emethod_TRAIN_NesterovWithHESSIANasG_AUC = []
	EmethodNesterovWithHESSIANasG_SIGMOID = []
	EmethodNesterovWithHESSIANasG_TIME = []

	EmethodNesterovWithHESSIANasG_TIME.append(Tend - Tbegin)
	NesterovAGwithHESSIANasG_TIME.write(str(Tend - Tbegin)); NesterovAGwithHESSIANasG_TIME.write(",")
	# log-likelihood
	MtestX = np.matrix(testX)
	newMtestXV = MtestX * MV
	loghx = []
	for idx in range(len(testY)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TEST_NesterovWithHESSIANasG_MLE.append(loglikelihood)
	test_NesterovAGwithHESSIANasG_MLE.write(str(loglikelihood)); test_NesterovAGwithHESSIANasG_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(testY)):
		hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TEST_NesterovWithHESSIANasG_AUC.append(ROCAUC(hxlist, testY))
	test_NesterovAGwithHESSIANasG_AUC.write(str(ROCAUC(hxlist, testY))); test_NesterovAGwithHESSIANasG_AUC.write(",")
	#---------------------------------------------------------------------#
	# log-likelihood
	MX = np.matrix(X)
	newMXW = MX * MV
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	Emethod_TRAIN_NesterovWithHESSIANasG_MLE.append(loglikelihood)
	train_NesterovAGwithHESSIANasG_MLE.write(str(loglikelihood)); train_NesterovAGwithHESSIANasG_MLE.write(",")
	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	Emethod_TRAIN_NesterovWithHESSIANasG_AUC.append(ROCAUC(hxlist, Y))
	train_NesterovAGwithHESSIANasG_AUC.write(str(ROCAUC(hxlist, Y))); train_NesterovAGwithHESSIANasG_AUC.write(",")

	alpha0 = 0.01
	alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0
	# Stage 3.
	#     Start the Gradient Descent algorithm
	#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	#           grad = [Y@(1 - sigm(yWTx))]T * X
	for iter in range(MAX_ITER):
		break
		Tbegin = time.clock()
		curSigmoidInput = []

	#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
		# W.T * X
		MXV = MX * MV
		# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
		yhypothesis = []
		for idx in range(len(Y)):

			curSigmoidInput.append(Y[idx]*MXV.A[idx][0])

			# hlambda(): the polynomial function to substitute the Sigmoid function
			h = 1 - hlambda(testx(Y[idx]*MXV.A[idx][0]))

			yhypothesis.append([h*Y[idx]])

		Myhypothesis = np.matrix(yhypothesis)
		# g = [Y@(1 - sigm(yWTx))]T * X	
		Mg = MXT * Myhypothesis

		# Construct the Hessian matrix = - MXT * MMID * MX
		hypothesis = [ [1.0/(1+exp(-testx(wTx)))] for wTx in [x[0] for x in MXV.A] ]
		MMID = np.matrix(np.eye(MX.shape[0]))
		for idx in range(MMID.shape[0]):
			MMID[idx,idx] = hypothesis[idx][0] * (1-hypothesis[idx][0])
		MHHES = -1 * MXT * MMID * MX

		MB2 = np.matrix(np.eye(MHHES.shape[0]))
		MB2_inv = np.matrix(np.eye(MHHES.shape[0]))
		for rowidx in range(MB2.shape[0]):
			temp = 0.0
			for colidx in range(MB2.shape[1]):
				temp += abs(MHHES[rowidx,colidx])
			MB2_inv[rowidx,rowidx] = 1.0/(temp+epsilon)

		eta = (1 - alpha0) / alpha1
		gamma = 1.0/(iter+1)/MX.shape[0]
		
		MG = MB2_inv * Mg          
		# should be 'plus', 'cause to compute the MLE
		MtmpW = MV + (gamma + 1.0) * MG           
		MV = (1.0-eta)*MtmpW + (eta)*MW
		MW = MtmpW

		alpha0 = alpha1
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0

		Tend = time.clock()
		#Ebonte_TIME.append(Tend - Tbegin + Ebonte_TIME[-1])
		EmethodNesterovWithHESSIANasG_TIME.append(Tend - Tbegin)
		NesterovAGwithHESSIANasG_TIME.write(str(Tend - Tbegin))
		if iter+1!=MAX_ITER : NesterovAGwithHESSIANasG_TIME.write(",")

		EmethodNesterovWithHESSIANasG_SIGMOID.append(curSigmoidInput)         
	#     Step 4. Calculate the cost function using Maximum likelihood Estimation
		# log-likelihood
		MtestX = np.matrix(testX)
		newMtestXV = MtestX * MV
		loghx = []
		for idx in range(len(testY)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(testY[idx]*newMtestXV.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TEST_NesterovWithHESSIANasG_MLE.append(loglikelihood)
		test_NesterovAGwithHESSIANasG_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : test_NesterovAGwithHESSIANasG_MLE.write(",")
		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(testY)):
			hx = 1.0/(1+exp(-testx(newMtestXV.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TEST_NesterovWithHESSIANasG_AUC.append(ROCAUC(hxlist, testY))
		test_NesterovAGwithHESSIANasG_AUC.write(str(ROCAUC(hxlist, testY)))
		if iter+1!=MAX_ITER : test_NesterovAGwithHESSIANasG_AUC.write(",")
		#print iter, '-th AUC : ', ROCAUC(hxlist, testY), ' MLE = ', loglikelihood
		

		#---------------------------------------------------------------------#
		# log-likelihood
		MX = np.matrix(X)
		newMXW = MX * MV
		loghx = []
		for idx in range(len(Y)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(-testx(Y[idx]*newMXW.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_TRAIN_NesterovWithHESSIANasG_MLE.append(loglikelihood)
		train_NesterovAGwithHESSIANasG_MLE.write(str(loglikelihood))
		if iter+1!=MAX_ITER : train_NesterovAGwithHESSIANasG_MLE.write(",")
		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(Y)):
			hx = 1.0/(1+exp(-testx(newMXW.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		Emethod_TRAIN_NesterovWithHESSIANasG_AUC.append(ROCAUC(hxlist, Y))
		train_NesterovAGwithHESSIANasG_AUC.write(str(ROCAUC(hxlist, Y)))
		if iter+1!=MAX_ITER : train_NesterovAGwithHESSIANasG_AUC.write(",")
		print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	'''
	-------------------------------------------------------------------------------------------
	----------------------------------- end of experiments -----------------------------------
	-------------------------------------------------------------------------------------------
	'''

	'''---------------------------- BonteSFH ----------------------------'''
	test_BonteSFH_MLE.write("\n");	test_BonteSFH_AUC.write("\n"); 
	train_BonteSFH_MLE.write("\n");	train_BonteSFH_AUC.write("\n"); 	
	BonteSFH_TIME.write("\n");
	'''---------------------- BonteSFH + .25XTXasG ----------------------'''
	test_BonteSFHwithXTXasG_MLE.write("\n");	test_BonteSFHwithXTXasG_AUC.write("\n");
	train_BonteSFHwithXTXasG_MLE.write("\n");	train_BonteSFHwithXTXasG_AUC.write("\n");
	BonteSFHwithXTXasG_TIME.write("\n"); 
	'''---------------------- BonteSFH + HESSIANasG ---------------------'''
	test_BonteSFHwithHESSIAN_MLE.write("\n");	test_BonteSFHwithHESSIAN_AUC.write("\n"); 
	train_BonteSFHwithHESSIAN_MLE.write("\n"); 	train_BonteSFHwithHESSIAN_AUC.write("\n");
	BonteSFHwithHESSIAN_TIME.write("\n"); 
	'''--------------------------- NesterovAG ---------------------------'''
	test_NesterovAG_MLE.write("\n");	test_NesterovAG_AUC.write("\n");
	train_NesterovAG_MLE.write("\n"); 	train_NesterovAG_AUC.write("\n"); 
	NesterovAG_TIME.write("\n"); 
	'''--------------------- NesterovAG + .25XTXasG ---------------------'''
	test_NesterovAGwithXTXasG_MLE.write("\n"); 	test_NesterovAGwithXTXasG_AUC.write("\n"); 
	train_NesterovAGwithXTXasG_MLE.write("\n");	train_NesterovAGwithXTXasG_AUC.write("\n");
	NesterovAGwithXTXasG_TIME.write("\n"); 
	'''---------------------- NesterovAG + HESSIAN ----------------------'''
	test_NesterovAGwithHESSIANasG_MLE.write("\n"); 	test_NesterovAGwithHESSIANasG_AUC.write("\n");
	train_NesterovAGwithHESSIANasG_MLE.write("\n");	train_NesterovAGwithHESSIANasG_AUC.write("\n");
	NesterovAGwithHESSIANasG_TIME.write("\n");


	label = [ 'BonteSFH', 'BonteSFH + .25XTXasG',  'NesterovAG', 'NesterovAG + .25XTXasG' ]
	plt.plot(range(len(Emethod_TEST_Bonte_MLE)), Emethod_TEST_Bonte_MLE, 'o--b')
	plt.plot(range(len(Emethod_TEST_BonteWithXTXasG_MLE)), Emethod_TEST_BonteWithXTXasG_MLE, 'v--b')
	plt.plot(range(len(Emethod_TEST_Nesterov_MLE)), Emethod_TEST_Nesterov_MLE, 'o--r')
	plt.plot(range(len(Emethod_TEST_NesterovWithXTXasG_MLE)), Emethod_TEST_NesterovWithXTXasG_MLE, '^--r')
	#plt.axis("equal")
	plt.title(str_idxcv+ '_TEST_MLE')
	plt.legend(label, loc = 0, ncol = 1)  
	plt.grid()
	plt.show()
	#plt.savefig("MLE4.png")
	#plt.close()

	plt.plot(range(len(Emethod_TRAIN_Bonte_MLE)), Emethod_TRAIN_Bonte_MLE, 'o--b')
	plt.plot(range(len(Emethod_TRAIN_BonteWithXTXasG_MLE)), Emethod_TRAIN_BonteWithXTXasG_MLE, 'v--b')
	plt.plot(range(len(Emethod_TRAIN_Nesterov_MLE)), Emethod_TRAIN_Nesterov_MLE, 'o--r')
	plt.plot(range(len(Emethod_TRAIN_NesterovWithXTXasG_MLE)), Emethod_TRAIN_NesterovWithXTXasG_MLE, '^--r')
	#plt.axis("equal")
	plt.title(str_idxcv+ '_TRAIN_MLE')
	plt.legend(label, loc = 0, ncol = 1)  
	plt.grid()
	plt.show()
	#plt.savefig("MLE4.png")
	#plt.close()

	plt.plot(range(len(Emethod_TEST_Bonte_AUC)), Emethod_TEST_Bonte_AUC, 'o--b')
	plt.plot(range(len(Emethod_TEST_BonteWithXTXasG_AUC)), Emethod_TEST_BonteWithXTXasG_AUC, 'v--b')
	plt.plot(range(len(Emethod_TEST_Nesterov_AUC)), Emethod_TEST_Nesterov_AUC, 'o--r')
	plt.plot(range(len(Emethod_TEST_NesterovWithXTXasG_AUC)), Emethod_TEST_NesterovWithXTXasG_AUC, '^--r')
	plt.title(str_idxcv+ '_TEST_AUC')
	plt.legend(label, loc = 0, ncol = 1)  
	plt.grid()
	plt.show()
	#plt.savefig("AUC4.png")
	#plt.close()

	plt.plot(range(len(Emethod_TRAIN_Bonte_AUC)), Emethod_TRAIN_Bonte_AUC, 'o--b')
	plt.plot(range(len(Emethod_TRAIN_BonteWithXTXasG_AUC)), Emethod_TRAIN_BonteWithXTXasG_AUC, 'v--b')
	plt.plot(range(len(Emethod_TRAIN_Nesterov_AUC)), Emethod_TRAIN_Nesterov_AUC, 'o--r')
	plt.plot(range(len(Emethod_TRAIN_NesterovWithXTXasG_AUC)), Emethod_TRAIN_NesterovWithXTXasG_AUC, '^--r')
	plt.title(str_idxcv+ '_TRAIN_AUC')
	plt.legend(label, loc = 0, ncol = 1)  
	plt.grid()
	plt.show()
	#plt.savefig("AUC4.png")
	#plt.close()


	plt.plot(range(len(EmethodBonte_TIME)), EmethodBonte_TIME, 'o--b')
	plt.plot(range(len(EmethodBonteWithXTXasG_TIME)), EmethodBonteWithXTXasG_TIME, 'v--b')
	plt.plot(range(len(EmethodNesterov_TIME)), EmethodNesterov_TIME, 'o--r')
	plt.plot(range(len(EmethodNesterovWithXTXasG_TIME)), EmethodNesterovWithXTXasG_TIME, '^--r')
	plt.title(str_idxcv+ 'TIME')
	plt.legend(label, loc = 0, ncol = 1)  
	plt.grid()
	plt.show()
	#plt.savefig("AUC4.png")
	#plt.close()


	plt.close()
	for iter in range(len(EmethodBonte_SIGMOID)):
		plt.plot([iter]*len(EmethodBonte_SIGMOID[iter]), EmethodBonte_SIGMOID[iter], '.-b')
		miny = min(EmethodBonte_SIGMOID[iter])
		maxy = max(EmethodBonte_SIGMOID[iter])
		plt.text(iter, miny, '%.3f' % miny, ha='center', va='top', fontsize=10)
		plt.text(iter, maxy, '%.3f' % maxy, ha='center', va='bottom', fontsize=10) 
		print '[ ',min(EmethodBonte_SIGMOID[iter]), ' , ', max(EmethodBonte_SIGMOID[iter]), ' ]  '
	plt.title(str_idxcv+ 'INPUT RANGE OF SIGMOID : Bonte')
	plt.grid()
	plt.show()

	plt.close()
	for iter in range(len(EmethodNesterovWithXTXasG_SIGMOID)):
		plt.plot([iter]*len(EmethodNesterovWithXTXasG_SIGMOID[iter]), EmethodNesterovWithXTXasG_SIGMOID[iter], '.-b')
		miny = min(EmethodNesterovWithXTXasG_SIGMOID[iter])
		maxy = max(EmethodNesterovWithXTXasG_SIGMOID[iter])
		plt.text(iter, miny, '%.3f' % miny, ha='center', va='top', fontsize=10)
		plt.text(iter, maxy, '%.3f' % maxy, ha='center', va='bottom', fontsize=10) 
		print '[ ',min(EmethodNesterovWithXTXasG_SIGMOID[iter]), ' , ', max(EmethodNesterovWithXTXasG_SIGMOID[iter]), ' ]  '
	plt.title(str_idxcv+ 'INPUT RANGE OF SIGMOID : Bonte + HESSIANasG')
	plt.grid()
	plt.show()


'''---------------------------- BonteSFH ----------------------------'''
test_BonteSFH_MLE.close();	test_BonteSFH_AUC.close(); 
train_BonteSFH_MLE.close();	train_BonteSFH_AUC.close(); 	
BonteSFH_TIME.close();
'''---------------------- BonteSFH + .25XTXasG ----------------------'''
test_BonteSFHwithXTXasG_MLE.close();	test_BonteSFHwithXTXasG_AUC.close();
train_BonteSFHwithXTXasG_MLE.close();	train_BonteSFHwithXTXasG_AUC.close();
BonteSFHwithXTXasG_TIME.close(); 
'''---------------------- BonteSFH + HESSIANasG ---------------------'''
test_BonteSFHwithHESSIAN_MLE.close();	test_BonteSFHwithHESSIAN_AUC.close(); 
train_BonteSFHwithHESSIAN_MLE.close(); 	train_BonteSFHwithHESSIAN_AUC.close();
BonteSFHwithHESSIAN_TIME.close(); 
'''--------------------------- NesterovAG ---------------------------'''
test_NesterovAG_MLE.close();	test_NesterovAG_AUC.close();
train_NesterovAG_MLE.close(); 	train_NesterovAG_AUC.close(); 
NesterovAG_TIME.close();
'''--------------------- NesterovAG + .25XTXasG ---------------------'''
test_NesterovAGwithXTXasG_MLE.close(); 	test_NesterovAGwithXTXasG_AUC.close(); 
train_NesterovAGwithXTXasG_MLE.close();	train_NesterovAGwithXTXasG_AUC.close();
NesterovAGwithXTXasG_TIME.close(); 
'''---------------------- NesterovAG + HESSIAN ----------------------'''
test_NesterovAGwithHESSIANasG_MLE.close(); 	test_NesterovAGwithHESSIANasG_AUC.close();
train_NesterovAGwithHESSIANasG_MLE.close();	train_NesterovAGwithHESSIANasG_AUC.close();
NesterovAGwithHESSIANasG_TIME.close();



'''---------------------------- DRAW THE FIVE CV AVERAGE RESULT ----------------------------'''
'''---------------------------- BonteSFH ----------------------------'''
test_BonteSFH_MLE =  open(resfilepath_test_BonteSFH_MLE,  'r')
test_BonteSFH_AUC =  open(resfilepath_test_BonteSFH_AUC,  'r')
train_BonteSFH_MLE = open(resfilepath_train_BonteSFH_MLE, 'r')
train_BonteSFH_AUC = open(resfilepath_train_BonteSFH_AUC, 'r')
BonteSFH_TIME =      open(resfilepath_BonteSFH_TIME,      'r')
'''---------------------- BonteSFH + .25XTXasG ----------------------'''
test_BonteSFHwithXTXasG_MLE =  open(resfilepath_test_BonteSFHwithXTXasG_MLE,  'r')
test_BonteSFHwithXTXasG_AUC =  open(resfilepath_test_BonteSFHwithXTXasG_AUC,  'r')
train_BonteSFHwithXTXasG_MLE = open(resfilepath_train_BonteSFHwithXTXasG_MLE, 'r')
train_BonteSFHwithXTXasG_AUC = open(resfilepath_train_BonteSFHwithXTXasG_AUC, 'r')
BonteSFHwithXTXasG_TIME =      open(resfilepath_BonteSFHwithXTXasG_TIME,      'r')
'''---------------------- BonteSFH + HESSIANasG ---------------------'''
test_BonteSFHwithHESSIAN_MLE =  open(resfilepath_test_BonteSFHwithHESSIAN_MLE,  'r')
test_BonteSFHwithHESSIAN_AUC =  open(resfilepath_test_BonteSFHwithHESSIAN_AUC,  'r')
train_BonteSFHwithHESSIAN_MLE = open(resfilepath_train_BonteSFHwithHESSIAN_MLE, 'r')
train_BonteSFHwithHESSIAN_AUC = open(resfilepath_train_BonteSFHwithHESSIAN_AUC, 'r')
BonteSFHwithHESSIAN_TIME =      open(resfilepath_BonteSFHwithHESSIAN_TIME,      'r')
'''--------------------------- NesterovAG ---------------------------'''
test_NesterovAG_MLE =  open(resfilepath_test_NesterovAG_MLE,  'r')
test_NesterovAG_AUC =  open(resfilepath_test_NesterovAG_AUC,  'r')
train_NesterovAG_MLE = open(resfilepath_train_NesterovAG_MLE, 'r')
train_NesterovAG_AUC = open(resfilepath_train_NesterovAG_AUC, 'r')
NesterovAG_TIME =      open(resfilepath_NesterovAG_TIME,      'r')
'''--------------------- NesterovAG + .25XTXasG ---------------------'''
test_NesterovAGwithXTXasG_MLE =  open(resfilepath_test_NesterovAGwithXTXasG_MLE,  'r')
test_NesterovAGwithXTXasG_AUC =  open(resfilepath_test_NesterovAGwithXTXasG_AUC,  'r')
train_NesterovAGwithXTXasG_MLE = open(resfilepath_train_NesterovAGwithXTXasG_MLE, 'r')
train_NesterovAGwithXTXasG_AUC = open(resfilepath_train_NesterovAGwithXTXasG_AUC, 'r')
NesterovAGwithXTXasG_TIME =      open(resfilepath_NesterovAGwithXTXasG_TIME,      'r')
'''---------------------- NesterovAG + HESSIAN ----------------------'''
test_NesterovAGwithHESSIANasG_MLE =  open(resfilepath_test_NesterovAGwithHESSIANasG_MLE,  'r')
test_NesterovAGwithHESSIANasG_AUC =  open(resfilepath_test_NesterovAGwithHESSIANasG_AUC,  'r')
train_NesterovAGwithHESSIANasG_MLE = open(resfilepath_train_NesterovAGwithHESSIANasG_MLE, 'r')
train_NesterovAGwithHESSIANasG_AUC = open(resfilepath_train_NesterovAGwithHESSIANasG_AUC, 'r')
NesterovAGwithHESSIANasG_TIME =      open(resfilepath_NesterovAGwithHESSIANasG_TIME,      'r')


# Calculate the average CV result from the csv file
# WARNNING: The First Element of Each Line From csvfile is a descriptive string like "CV-0TH: "
def getdata(csvfile):
	reader = csv.reader(csvfile)
	data = []
	for row in reader:
		row = [float(x) for x in row[1:]]
		data.append(row)
	#print 'data = ', data
	res = [0]*len(data[0])	
	for row in data:
		for idx in range(len(res)):
			res[idx] += row[idx]
	for idx in range(len(res)):
		res[idx] /= len(data)

	csvfile.close()
	return res

'''---------------------------- BonteSFH ----------------------------'''
Emethod_TEST_Bonte_MLE = getdata(test_BonteSFH_MLE)
Emethod_TEST_Bonte_AUC = getdata(test_BonteSFH_AUC) 
Emethod_TRAIN_Bonte_MLE = getdata(train_BonteSFH_MLE)
Emethod_TRAIN_Bonte_AUC = getdata(train_BonteSFH_AUC)	
EmethodBonte_TIME = getdata(BonteSFH_TIME)
'''---------------------- BonteSFH + .25XTXasG ----------------------'''
Emethod_TEST_BonteWithXTXasG_MLE = getdata(test_BonteSFHwithXTXasG_MLE)
Emethod_TEST_BonteWithXTXasG_AUC = getdata(test_BonteSFHwithXTXasG_AUC)
Emethod_TRAIN_BonteWithXTXasG_MLE = getdata(train_BonteSFHwithXTXasG_MLE)	
Emethod_TRAIN_BonteWithXTXasG_AUC = getdata(train_BonteSFHwithXTXasG_AUC)
EmethodBonteWithXTXasG_TIME = getdata(BonteSFHwithXTXasG_TIME)
'''---------------------- BonteSFH + HESSIANasG ---------------------'''
'''--------------------------- NesterovAG ---------------------------'''
Emethod_TEST_Nesterov_MLE = getdata(test_NesterovAG_MLE)	
Emethod_TEST_Nesterov_AUC = getdata(test_NesterovAG_AUC)
Emethod_TRAIN_Nesterov_MLE = getdata(train_NesterovAG_MLE) 	
Emethod_TRAIN_Nesterov_AUC = getdata(train_NesterovAG_AUC) 
EmethodNesterov_TIME = getdata(NesterovAG_TIME)
'''--------------------- NesterovAG + .25XTXasG ---------------------'''
Emethod_TEST_NesterovWithXTXasG_MLE = getdata(test_NesterovAGwithXTXasG_MLE) 	
Emethod_TEST_NesterovWithXTXasG_AUC = getdata(test_NesterovAGwithXTXasG_AUC)
Emethod_TRAIN_NesterovWithXTXasG_MLE = getdata(train_NesterovAGwithXTXasG_MLE)	
Emethod_TRAIN_NesterovWithXTXasG_AUC = getdata(train_NesterovAGwithXTXasG_AUC)
EmethodNesterovWithXTXasG_TIME = getdata(NesterovAGwithXTXasG_TIME)
'''---------------------- NesterovAG + HESSIAN ----------------------'''





'''------------------------------------------------------------------------------------------------------'''
label = [ 'SFH', 'SFH + G', 'NAG', 'NAG + G'  ]
plt.plot(range(len(Emethod_TEST_Bonte_MLE)), Emethod_TEST_Bonte_MLE, 'o--b')
plt.plot(range(len(Emethod_TEST_BonteWithXTXasG_MLE)), Emethod_TEST_BonteWithXTXasG_MLE, 'v--b')
plt.plot(range(len(Emethod_TEST_Nesterov_MLE)), Emethod_TEST_Nesterov_MLE, 'o--r')
plt.plot(range(len(Emethod_TEST_NesterovWithXTXasG_MLE)), Emethod_TEST_NesterovWithXTXasG_MLE, '^--r')
#plt.axis("equal")
#plt.title('Five-CV _TEST_MLE')
plt.xlabel("Iteration Number")
plt.ylabel("Maximum Log-likelihood Estimation")
plt.xlim([1, len(Emethod_TEST_Bonte_MLE)-1])
plt.legend(label, loc = 4, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("MLE4.png")
#plt.close()
plt.plot(range(len(Emethod_TRAIN_Bonte_MLE)), Emethod_TRAIN_Bonte_MLE, 'o--b')
plt.plot(range(len(Emethod_TRAIN_BonteWithXTXasG_MLE)), Emethod_TRAIN_BonteWithXTXasG_MLE, 'v--b')
plt.plot(range(len(Emethod_TRAIN_Nesterov_MLE)), Emethod_TRAIN_Nesterov_MLE, 'o--r')
plt.plot(range(len(Emethod_TRAIN_NesterovWithXTXasG_MLE)), Emethod_TRAIN_NesterovWithXTXasG_MLE, '^--r')
#plt.axis("equal")
#plt.title('Five_Cross_Validation_TRAIN_MLE')
plt.xlabel("Iteration Number")
plt.ylabel("Maximum Log-likelihood Estimation")
plt.xlim([1, len(Emethod_TRAIN_Bonte_MLE)-1])
plt.legend(label, loc = 4, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("MLE4.png")
#plt.close()
plt.plot(range(len(Emethod_TEST_Bonte_AUC[1:])), Emethod_TEST_Bonte_AUC[1:], 'o--b')
plt.plot(range(len(Emethod_TEST_BonteWithXTXasG_AUC[1:])), Emethod_TEST_BonteWithXTXasG_AUC[1:], 'v--b')
plt.plot(range(len(Emethod_TEST_Nesterov_AUC[1:])), Emethod_TEST_Nesterov_AUC[1:], 'o--r')
plt.plot(range(len(Emethod_TEST_NesterovWithXTXasG_AUC[1:])), Emethod_TEST_NesterovWithXTXasG_AUC[1:], '^--r')
#plt.title('Five_Cross_Validation_TEST_AUC')
plt.xlabel("Iteration Number")
plt.ylabel("Area Under the Curve")
plt.xlim([1, len(Emethod_TEST_Bonte_AUC)-1])
plt.legend(label, loc = 4, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("AUC4.png")
#plt.close()
plt.plot(range(len(Emethod_TRAIN_Bonte_AUC[1:])), Emethod_TRAIN_Bonte_AUC[1:], 'o--b')
plt.plot(range(len(Emethod_TRAIN_BonteWithXTXasG_AUC[1:])), Emethod_TRAIN_BonteWithXTXasG_AUC[1:], 'v--b')
plt.plot(range(len(Emethod_TRAIN_Nesterov_AUC[1:])), Emethod_TRAIN_Nesterov_AUC[1:], 'o--r')
plt.plot(range(len(Emethod_TRAIN_NesterovWithXTXasG_AUC[1:])), Emethod_TRAIN_NesterovWithXTXasG_AUC[1:], '^--r')
#plt.title('Five_Cross_Validation_TRAIN_AUC')
plt.xlabel("Iteration Number")
plt.ylabel("Area Under the Curve")
plt.xlim([1, len(Emethod_TRAIN_Bonte_AUC)-1])
plt.legend(label, loc = 4, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("AUC4.png")
#plt.close()
plt.plot(range(len(EmethodBonte_TIME)), EmethodBonte_TIME, 'o--b')
plt.plot(range(len(EmethodBonteWithXTXasG_TIME)), EmethodBonteWithXTXasG_TIME, 'v--b')
plt.plot(range(len(EmethodNesterov_TIME)), EmethodNesterov_TIME, 'o--r')
plt.plot(range(len(EmethodNesterovWithXTXasG_TIME)), EmethodNesterovWithXTXasG_TIME, '^--r')
plt.title('Five_Cross_Validation_TIME')
plt.legend(label, loc = 0, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("AUC4.png")
#plt.close()
'''------------------------------------------------------------------------------------------------------'''


'''---------------------------- Write the Five CV Result Into CSV Files ----------------------------'''
'''--------- so that we can plot the result in latex or overleaf with the data in csvfiles ---------'''
resfilepath_test_MLE =  str_dataset + 'FiveCV TEST MLE.csv'
resfilepath_test_AUC =  str_dataset + 'FiveCV TEST AUC.csv'
resfilepath_train_MLE = str_dataset + 'FiveCV TRAIN MLE.csv'
resfilepath_train_AUC = str_dataset + 'FiveCV TRAIN AUC.csv'
resfilepath_TIME =      str_dataset + 'FiveCV TIME.csv'
FiveCV_test_MLE =  open(resfilepath_test_MLE,  'w')
FiveCV_test_AUC =  open(resfilepath_test_AUC,  'w')
FiveCV_train_MLE = open(resfilepath_train_MLE, 'w')
FiveCV_train_AUC = open(resfilepath_train_AUC, 'w')
FiveCV_TIME =      open(resfilepath_TIME,      'w')

FiveCV_test_MLE =  open(resfilepath_test_MLE,  'a+b')
FiveCV_test_AUC =  open(resfilepath_test_AUC,  'a+b')
FiveCV_train_MLE = open(resfilepath_train_MLE, 'a+b')
FiveCV_train_AUC = open(resfilepath_train_AUC, 'a+b')
FiveCV_TIME =      open(resfilepath_TIME,      'a+b')

# Write the average CV result into the csv file
# WARNNING: Just write into the data without the descriptive string and write("\n") + close()
def writedata(fivecvres, csvfile):
	MAX_ITER = len(fivecvres)
	for (idx,ele) in enumerate(fivecvres):
		csvfile.write(str(ele))
		if idx+1 != MAX_ITER : 
			csvfile.write(',')

'''---------------------------- Write the Five CV Result Into CSV Files ----------------------------'''
'''-------- write the descriptive word into the first element of each row of every csv file --------'''
FiveCV_test_MLE.write("Emethod_TEST_Bonte_MLE");  FiveCV_test_MLE.write(',')
writedata(Emethod_TEST_Bonte_MLE, FiveCV_test_MLE);  FiveCV_test_MLE.write("\n");  
FiveCV_test_MLE.write("Emethod_TEST_BonteWithXTXasG_MLE");  FiveCV_test_MLE.write(',')
writedata(Emethod_TEST_BonteWithXTXasG_MLE, FiveCV_test_MLE);  FiveCV_test_MLE.write("\n");  
FiveCV_test_MLE.write("Emethod_TEST_BonteWithHESSIANasG_MLE");  FiveCV_test_MLE.write(',')
writedata(Emethod_TEST_BonteWithHESSIANasG_MLE, FiveCV_test_MLE);  FiveCV_test_MLE.write("\n"); 
FiveCV_test_MLE.write("Emethod_TEST_Nesterov_MLE");  FiveCV_test_MLE.write(',')
writedata(Emethod_TEST_Nesterov_MLE, FiveCV_test_MLE);  FiveCV_test_MLE.write("\n"); 
FiveCV_test_MLE.write("Emethod_TEST_NesterovWithXTXasG_MLE");  FiveCV_test_MLE.write(',')
writedata(Emethod_TEST_NesterovWithXTXasG_MLE, FiveCV_test_MLE);  FiveCV_test_MLE.write("\n");  
FiveCV_test_MLE.write("Emethod_TEST_NesterovWithHESSIANasG_MLE");  FiveCV_test_MLE.write(',')
writedata(Emethod_TEST_NesterovWithHESSIANasG_MLE, FiveCV_test_MLE);  FiveCV_test_MLE.write("\n"); 
FiveCV_test_MLE.close()

FiveCV_test_AUC.write("Emethod_TEST_Bonte_AUC");  FiveCV_test_AUC.write(',')
writedata(Emethod_TEST_Bonte_AUC, FiveCV_test_AUC);  FiveCV_test_AUC.write("\n");  
FiveCV_test_AUC.write("Emethod_TEST_BonteWithXTXasG_AUC");  FiveCV_test_AUC.write(',')
writedata(Emethod_TEST_BonteWithXTXasG_AUC, FiveCV_test_AUC);  FiveCV_test_AUC.write("\n");  
FiveCV_test_AUC.write("Emethod_TEST_BonteWithHESSIANasG_AUC");  FiveCV_test_AUC.write(',')
writedata(Emethod_TEST_BonteWithHESSIANasG_AUC, FiveCV_test_AUC);  FiveCV_test_AUC.write("\n");  
FiveCV_test_AUC.write("Emethod_TEST_Nesterov_AUC");  FiveCV_test_AUC.write(',')
writedata(Emethod_TEST_Nesterov_AUC, FiveCV_test_AUC);  FiveCV_test_AUC.write("\n");  
FiveCV_test_AUC.write("Emethod_TEST_NesterovWithXTXasG_AUC");  FiveCV_test_AUC.write(',')
writedata(Emethod_TEST_NesterovWithXTXasG_AUC, FiveCV_test_AUC);  FiveCV_test_AUC.write("\n");  
FiveCV_test_AUC.write("Emethod_TEST_NesterovWithHESSIANasG_AUC");  FiveCV_test_AUC.write(',')
writedata(Emethod_TEST_NesterovWithHESSIANasG_AUC, FiveCV_test_AUC);  FiveCV_test_AUC.write("\n");  
FiveCV_test_AUC.close()


FiveCV_train_MLE.write("Emethod_TRAIN_Bonte_MLE");  FiveCV_train_MLE.write(',')
writedata(Emethod_TRAIN_Bonte_MLE, FiveCV_train_MLE);  FiveCV_train_MLE.write("\n");  
FiveCV_train_MLE.write("Emethod_TRAIN_BonteWithXTXasG_MLE");  FiveCV_train_MLE.write(',')
writedata(Emethod_TRAIN_BonteWithXTXasG_MLE, FiveCV_train_MLE);  FiveCV_train_MLE.write("\n");  
FiveCV_train_MLE.write("Emethod_TRAIN_BonteWithHESSIANasG_MLE");  FiveCV_train_MLE.write(',')
writedata(Emethod_TRAIN_BonteWithHESSIANasG_MLE, FiveCV_train_MLE);  FiveCV_train_MLE.write("\n");  
FiveCV_train_MLE.write("Emethod_TRAIN_Nesterov_MLE");  FiveCV_train_MLE.write(',')
writedata(Emethod_TRAIN_Nesterov_MLE, FiveCV_train_MLE);  FiveCV_train_MLE.write("\n");  
FiveCV_train_MLE.write("Emethod_TRAIN_NesterovWithXTXasG_MLE");  FiveCV_train_MLE.write(',')
writedata(Emethod_TRAIN_NesterovWithXTXasG_MLE, FiveCV_train_MLE);  FiveCV_train_MLE.write("\n"); 
FiveCV_train_MLE.write("Emethod_TRAIN_NesterovWithHESSIANasG_MLE");  FiveCV_train_MLE.write(',')
writedata(Emethod_TRAIN_NesterovWithHESSIANasG_MLE, FiveCV_train_MLE);  FiveCV_train_MLE.write("\n");  
FiveCV_train_MLE.close()


FiveCV_train_AUC.write("Emethod_TRAIN_Bonte_AUC");  FiveCV_train_AUC.write(',')
writedata(Emethod_TRAIN_Bonte_AUC, FiveCV_train_AUC);  FiveCV_train_AUC.write("\n");  
FiveCV_train_AUC.write("Emethod_TRAIN_BonteWithXTXasG_AUC");  FiveCV_train_AUC.write(',')
writedata(Emethod_TRAIN_BonteWithXTXasG_AUC, FiveCV_train_AUC);  FiveCV_train_AUC.write("\n");  
FiveCV_train_AUC.write("Emethod_TRAIN_BonteWithHESSIANasG_AUC");  FiveCV_train_AUC.write(',')
writedata(Emethod_TRAIN_BonteWithHESSIANasG_AUC, FiveCV_train_AUC);  FiveCV_train_AUC.write("\n");  
FiveCV_train_AUC.write("Emethod_TRAIN_Nesterov_AUC");  FiveCV_train_AUC.write(',')
writedata(Emethod_TRAIN_Nesterov_AUC, FiveCV_train_AUC);  FiveCV_train_AUC.write("\n");  
FiveCV_train_AUC.write("Emethod_TRAIN_NesterovWithXTXasG_AUC");  FiveCV_train_AUC.write(',')
writedata(Emethod_TRAIN_NesterovWithXTXasG_AUC, FiveCV_train_AUC);  FiveCV_train_AUC.write("\n"); 
FiveCV_train_AUC.write("Emethod_TRAIN_NesterovWithHESSIANasG_AUC");  FiveCV_train_AUC.write(',')
writedata(Emethod_TRAIN_NesterovWithHESSIANasG_AUC, FiveCV_train_AUC);  FiveCV_train_AUC.write("\n");  
FiveCV_train_AUC.close()


FiveCV_TIME.write("EmethodBonte_TIME");  FiveCV_TIME.write(',')
writedata(EmethodBonte_TIME, FiveCV_TIME);  FiveCV_TIME.write("\n");  
FiveCV_TIME.write("EmethodBonteWithXTXasG_TIME");  FiveCV_TIME.write(',')
writedata(EmethodBonteWithXTXasG_TIME, FiveCV_TIME);  FiveCV_TIME.write("\n"); 
FiveCV_TIME.write("EmethodBonteWithHESSIANasG_TIME");  FiveCV_TIME.write(',')
writedata(EmethodBonteWithHESSIANasG_TIME, FiveCV_TIME);  FiveCV_TIME.write("\n");  
FiveCV_TIME.write("EmethodNesterov_TIME");  FiveCV_TIME.write(',')
writedata(EmethodNesterov_TIME, FiveCV_TIME);  FiveCV_TIME.write("\n"); 
FiveCV_TIME.write("EmethodNesterovWithXTXasG_TIME");  FiveCV_TIME.write(',')
writedata(EmethodNesterovWithXTXasG_TIME, FiveCV_TIME);  FiveCV_TIME.write("\n");  
FiveCV_TIME.write("EmethodNesterovWithHESSIANasG_TIME");  FiveCV_TIME.write(',')
writedata(EmethodNesterovWithHESSIANasG_TIME, FiveCV_TIME);  FiveCV_TIME.write("\n");  
FiveCV_TIME.close()



# --------------- FILE: TRAIN MLE -------------- 
# -- Iterations -- SFH -- SFHG -- NAG -- NAGG -- 
filePath = 'ICANN2020_PythonExperiment_2017iDASH_FiveCV_SFHvs.SFHGvs.NAGvs.NAGG_TRAIN_MLE.csv';
PythonExperimentMNIST =      open(filePath,      'w')
PythonExperimentMNIST =      open(filePath,      'a+b')

PythonExperimentMNIST.write('Iterations'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('SFH');   
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('SFHG'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAG'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAGG');   
PythonExperimentMNIST.write("\n");

#Emethod_TRAIN_Bonte_MLE   Emethod_TRAIN_BonteWithXTXasG_MLE
#Emethod_TRAIN_Nesterov_MLE  Emethod_TRAIN_NesterovWithXTXasG_MLE

for (idx, ele) in enumerate(Emethod_TRAIN_Bonte_MLE):
	PythonExperimentMNIST.write(str(idx)); 
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TRAIN_Bonte_MLE[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TRAIN_BonteWithXTXasG_MLE[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TRAIN_Nesterov_MLE[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TRAIN_NesterovWithXTXasG_MLE[idx]));
	PythonExperimentMNIST.write("\n");

PythonExperimentMNIST.close();


# ---------------- FILE: TEST MLE --------------- 
# -- Iterations -- SFH -- SFHG -- NAG -- NAGG -- 
filePath = 'ICANN2020_PythonExperiment_2017iDASH_FiveCV_SFHvs.SFHGvs.NAGvs.NAGG_TEST_MLE.csv';
PythonExperimentMNIST =      open(filePath,      'w')
PythonExperimentMNIST =      open(filePath,      'a+b')

PythonExperimentMNIST.write('Iterations'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('SFH');   
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('SFHG'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAG'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAGG');   
PythonExperimentMNIST.write("\n");

#Emethod_TEST_Bonte_MLE   Emethod_TEST_BonteWithXTXasG_MLE
#Emethod_TEST_Nesterov_MLE  Emethod_TEST_NesterovWithXTXasG_MLE

for (idx, ele) in enumerate(Emethod_TEST_Bonte_MLE):
	PythonExperimentMNIST.write(str(idx)); 
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TEST_Bonte_MLE[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TEST_BonteWithXTXasG_MLE[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TEST_Nesterov_MLE[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TEST_NesterovWithXTXasG_MLE[idx]));
	PythonExperimentMNIST.write("\n");

PythonExperimentMNIST.close();

# --------------- FILE: TRAIN AUC -------------- 
# -- Iterations -- SFH -- SFHG -- NAG -- NAGG -- 
filePath = 'ICANN2020_PythonExperiment_2017iDASH_FiveCV_SFHvs.SFHGvs.NAGvs.NAGG_TRAIN_AUC.csv';
PythonExperimentMNIST =      open(filePath,      'w')
PythonExperimentMNIST =      open(filePath,      'a+b')

PythonExperimentMNIST.write('Iterations'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('SFH');   
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('SFHG'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAG'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAGG');   
PythonExperimentMNIST.write("\n");

#Emethod_TRAIN_Bonte_MLE   Emethod_TRAIN_BonteWithXTXasG_MLE
#Emethod_TRAIN_Nesterov_MLE  Emethod_TRAIN_NesterovWithXTXasG_MLE

for (idx, ele) in enumerate(Emethod_TRAIN_Bonte_AUC):
	PythonExperimentMNIST.write(str(idx)); 
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TRAIN_Bonte_AUC[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TRAIN_BonteWithXTXasG_AUC[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TRAIN_Nesterov_AUC[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TRAIN_NesterovWithXTXasG_AUC[idx]));
	PythonExperimentMNIST.write("\n");

PythonExperimentMNIST.close();


# ---------------- FILE: TEST MLE --------------- 
# -- Iterations -- SFH -- SFHG -- NAG -- NAGG -- 
filePath = 'ICANN2020_PythonExperiment_2017iDASH_FiveCV_SFHvs.SFHGvs.NAGvs.NAGG_TEST_AUC.csv';
PythonExperimentMNIST =      open(filePath,      'w')
PythonExperimentMNIST =      open(filePath,      'a+b')

PythonExperimentMNIST.write('Iterations'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('SFH');   
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('SFHG'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAG'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAGG');   
PythonExperimentMNIST.write("\n");

#Emethod_TEST_Bonte_MLE   Emethod_TEST_BonteWithXTXasG_MLE
#Emethod_TEST_Nesterov_MLE  Emethod_TEST_NesterovWithXTXasG_MLE

for (idx, ele) in enumerate(Emethod_TEST_Bonte_AUC):
	PythonExperimentMNIST.write(str(idx)); 
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TEST_Bonte_AUC[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TEST_BonteWithXTXasG_AUC[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TEST_Nesterov_AUC[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(Emethod_TEST_NesterovWithXTXasG_AUC[idx]));
	PythonExperimentMNIST.write("\n");

PythonExperimentMNIST.close();

