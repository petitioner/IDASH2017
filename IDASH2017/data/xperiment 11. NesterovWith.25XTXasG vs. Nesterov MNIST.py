# coding=utf8
# 2019-12-05 09:32 a.m. GMT +08：00
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
from math import log, exp, pow, sqrt

import matplotlib
#matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

MAX_ITER = 300
epsilon = 1e-08

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
print "------------- Experiment11. Nesterov With QG vs. Nesterov            -------------"
print '-------------    Data Set :                                          -------------'
print '-------------           X : [[1,x11,x12,...],[1,x21,x22,...],...]    -------------'
print '-------------           Y : y = {-1, +1}                             -------------'
print '----------------------------------------------------------------------------------'

import csv

# Stage 1. 
#     Step 1. Extract data from a csv file
with open('data103x1579.txt','r') as csvfile:
#with open('edin.txt','r') as csvfile:
#with open('lbw.txt','r') as csvfile:
#with open('nhanes3.txt','r') as csvfile:
#with open('pcs.txt','r') as csvfile:
#with open('uis.txt','r') as csvfile:
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
X = [[1]+row[1:] for row in data[:]]
for colidx in range(len(X[0])):
	colmax = X[0][colidx]
	colmin = X[0][colidx]
	for (rowidx, row) in enumerate(X):
		if row[colidx] > colmax :
			colmax = row[colidx]
		if row[colidx] < colmin :
			colmin = row[colidx]
	for (rowidx, row) in enumerate(X):
		if (colmax - colmin) < epsilon:
			row[colidx] = .5;
		else:
			row[colidx] = (row[colidx] - colmin) / (colmax - colmin)
Y = [int(row[0]) for row in data[:]]
# turn y{+0,+1} to y{-1,+1}
Y = [2*y-1 for y in Y]    # DONT FORGET THAT THE IDASH DATASET IS DIFFERENT FROM THE MNIST DATASET!


#random.shuffle(X)
#should shuffle [Y,X] together!
'''
Z = zip(Y,X)
random.shuffle(Z)
#should shuffle [Y,X] together!
X = [item[1] for item in Z]
Y = [item[0] for item in Z]
'''


hlambda = lambda x:1.0/(1+exp(-x))
#hlambda = lambda x:5.0000e-01  +1.7786e-01*x  -3.6943e-03*pow(x,3)  +3.6602e-05*pow(x,5)  -1.2344e-07*pow(x,7)


'''
-------------------------------------------------------------------------------------------
------------------------- The Presented Method: Nesterov + QG        ----------------------
-------------------------------------------------------------------------------------------
''' 

# Stage 2. 
#     Step 1. Initialize Simplified Fixed Hessian Matrix
MX = np.matrix(X)
MXT = MX.T
MXTMX = MXT.dot(MX)                 

# H = +1/4 * X.T * X         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# ADAGRAD : ϵ is a smoothing term that avoids division by zero (usually on the order of 1e−8)
# 'cause MB already has (-.25), in this case, don't use the Quadratic Gradient Descent Method 

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
	mb[idx,idx] = (+.25)*mxtmx[idx, 0]  + epsilon           
	print 'M[',idx,'][',idx,'] = ',mb[idx,idx]
# END  : Bonte's Specific Order On XTX

MB = mb    
print MB  
MB_inv = np.matrix(np.eye(mxtmx.shape[0]))
for idx in range(mxtmx.shape[0]):
	MB_inv[idx,idx] = 1.0/mb[idx,idx] 
# get the inverse of matrix MB in advance


#     Step 2. Initialize Weight Vector (n x 1)
# Setting the initial weight to 1 leads to a large input to sigmoid function,
# which would cause a big problem to this algorithm when using polynomial
# to substitute the sigmoid function. So, it is a good choice to set w = 0.

# [[0]... to make MW a column vector(matrix)
V = [[0.0] for x in range(MB.shape[0])]
W = [[0.0] for x in range(MB.shape[0])]
MV = np.matrix(V)
MW = np.matrix(W)

#     Step 2. Set the Maximum Iteration and Record each cost function
EmethodNesterovWith_MLE = []
EmethodNesterovWith_AUC = []
EmethodNesterovWith_SIGMOID = []

alpha0 = 0.01
alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0

# Stage 3.
#     Start the Gradient Descent algorithm
#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
#           grad = [Y@(1 - sigm(yWTx))]T * X
for iter in range(MAX_ITER):
	curSigmoidInput = []

#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
	# W.T * X
	MXV = MX * MV
	# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
	yhypothesis = []
	for idx in range(len(Y)):

		curSigmoidInput.append(Y[idx]*MXV.A[idx][0])

		# hlambda(): the polynomial function to substitute the Sigmoid function
		h = 1 - hlambda(Y[idx]*MXV.A[idx][0])

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


	EmethodNesterovWith_SIGMOID.append(curSigmoidInput)         
#     Step 4. Calculate the cost function using Maximum likelihood Estimation
	# log-likelihood
	MtestX = np.matrix(X)
	newMtestXV = MtestX * MV
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-Y[idx]*newMtestXV.A[idx][0]))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	EmethodNesterovWith_MLE.append(loglikelihood)


	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-newMtestXV.A[idx][0]))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	EmethodNesterovWith_AUC.append(ROCAUC(hxlist, Y))


'''
-------------------------------------------------------------------------------------------
----------------------------------- end of experiments -----------------------------------
-------------------------------------------------------------------------------------------
'''

'''
-------------------------------------------------------------------------------------------
------------------------- The Presented Method: Nesterov AG -------------------------------
-------------------------------------------------------------------------------------------
''' 

# Stage 2. 
#     Step 1. Initialize Simplified Fixed Hessian Matrix
MX = np.matrix(X)
MXT = MX.T
MXTMX = MXT.dot(MX)                 


# return a column vector whose each element is the sum of each row of MX 
mx = MX.sum(axis=1)
# return a column vector whose each element is the sum of each row of (X.T * X)
print mx
mxtmx = MX.T.dot(mx)
print mxtmx
mb = np.matrix(np.eye(mxtmx.shape[0]))
for idx in range(mxtmx.shape[0]):
	mb[idx,idx] = (+.25)*mxtmx[idx, 0]  + epsilon           
	print 'M[',idx,'][',idx,'] = ',mb[idx,idx]
# END  : Bonte's Specific Order On XTX

MB = mb    
print MB  
MB_inv = np.matrix(np.eye(mxtmx.shape[0]))
for idx in range(mxtmx.shape[0]):
	MB_inv[idx,idx] = 1.0/mb[idx,idx] 
# get the inverse of matrix MB in advance


#     Step 2. Initialize Weight Vector (n x 1)
# [[0]... to make MW a column vector(matrix)
V = [[0.0] for x in range(MB.shape[0])]
W = [[0.0] for x in range(MB.shape[0])]
MV = np.matrix(V)
MW = np.matrix(W)

#     Step 2. Set the Maximum Iteration and Record each cost function
EmethodNesterov_MLE = []
EmethodNesterov_AUC = []
EmethodNesterov_SIGMOID = []

alpha0 = 0.01
alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0

# Stage 3.
#     Start the Gradient Descent algorithm
for iter in range(MAX_ITER):
	curSigmoidInput = []

    #Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
	# W.T * X
	MXV = MX * MV
	# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
	yhypothesis = []
	for idx in range(len(Y)):

		curSigmoidInput.append(Y[idx]*MXV.A[idx][0])

		# hlambda(): the polynomial function to substitute the Sigmoid function
		h = 1 - hlambda(Y[idx]*MXV.A[idx][0])

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


	EmethodNesterov_SIGMOID.append(curSigmoidInput)         
#     Step 4. Calculate the cost function using Maximum likelihood Estimation
	# log-likelihood
	MtestX = np.matrix(X)
	newMtestXV = MtestX * MV
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-Y[idx]*newMtestXV.A[idx][0]))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	EmethodNesterov_MLE.append(loglikelihood)


	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-newMtestXV.A[idx][0]))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	EmethodNesterov_AUC.append(ROCAUC(hxlist, Y))


'''
-------------------------------------------------------------------------------------------
----------------------------------- end of experiments -----------------------------------
-------------------------------------------------------------------------------------------
'''

label = [ 'Nesterov + .25XTXasG', 'Nesterov' ]
plt.plot(range(len(EmethodNesterovWith_MLE)), EmethodNesterovWith_MLE, 'v-b')
plt.plot(range(len(EmethodNesterov_MLE)), EmethodNesterov_MLE, 'o--r')
#plt.axis("equal")
plt.title('MLE')
plt.legend(label, loc = 0, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("MLE4.png")
#plt.close()

plt.plot(range(len(EmethodNesterovWith_AUC)), EmethodNesterovWith_AUC, 'v-b')
plt.plot(range(len(EmethodNesterov_AUC)), EmethodNesterov_AUC, 'o--r')
plt.title('AUC')
plt.legend(label, loc = 0, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("AUC4.png")
#plt.close()

'''
plt.close()
for iter in range(len(EmethodNesterovWith_SIGMOID)):
	plt.plot([iter]*len(EmethodNesterovWith_SIGMOID[iter]), EmethodNesterovWith_SIGMOID[iter], 'v-b')
	miny = min(EmethodNesterovWith_SIGMOID[iter])
	maxy = max(EmethodNesterovWith_SIGMOID[iter])
	plt.text(iter, miny, '%.3f' % miny, ha='center', va='top', fontsize=10)
	plt.text(iter, maxy, '%.3f' % maxy, ha='center', va='bottom', fontsize=10) 
	print '[ ',min(EmethodNesterovWith_SIGMOID[iter]), ' , ', max(EmethodNesterovWith_SIGMOID[iter]), ' ]  '
plt.title('INPUT RANGE OF SIGMOID : Nesterov + .25XTXasG')
plt.grid()
plt.show()
'''


# -------------- FILE: MLE -------------- 
# -- Iterations -- NAG -- NAGG -- 
filePath = 'PythonExperiment_NAGvs.NAGG_data103x1579_MLE.csv';
#filePath = 'PythonExperiment_NAGvs.NAGG_edin_MLE.csv';
#filePath = 'PythonExperiment_NAGvs.NAGG_lbw_MLE.csv';
#filePath = 'PythonExperiment_NAGvs.NAGG_nhanes3_MLE.csv';
#filePath = 'PythonExperiment_NAGvs.NAGG_pcs_MLE.csv';
#filePath = 'PythonExperiment_NAGvs.NAGG_uis_MLE.csv';
PythonExperimentMNIST =      open(filePath,      'w')
PythonExperimentMNIST =      open(filePath,      'a+b')

PythonExperimentMNIST.write('Iterations'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('NAG');    
PythonExperimentMNIST.write(','); 
PythonExperimentMNIST.write('NAGG');   
PythonExperimentMNIST.write("\n");

for (idx, ele) in enumerate(EmethodNesterovWith_MLE):
	PythonExperimentMNIST.write(str(idx)); 
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(EmethodNesterov_MLE[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(EmethodNesterovWith_MLE[idx]));
	PythonExperimentMNIST.write("\n");

PythonExperimentMNIST.close();