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
print "------------- Experiment11. Bonte With Learning Rate vs. Bonte       -------------"
print '-------------    Data Set : MNIST                                    -------------'
print '-------------           X : [[1,x11,x12,...],[1,x21,x22,...],...]    -------------'
print '-------------           Y : y = {-1, +1}                             -------------'
print '----------------------------------------------------------------------------------'

import csv

# Stage 1. 
#     Step 1. Extract data from a csv file
with open('MNISTt10k3(+1)8(-1)with14x14.csv','r') as csvfile:
#with open('MNISTtrain3(+1)8(-1)with14x14.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	#reader.next() # leave behind the first row
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
	colmax = 1.0
	for (rowidx, row) in enumerate(X):
		if row[colidx] > colmax :
			colmax = row[colidx]
	for (rowidx, row) in enumerate(X):
		row[colidx] /= colmax
Y = [int(row[0]) for row in data[:]]
# turn y{+0,+1} to y{-1,+1}
#Y = [2*y-1 for y in Y]    # DONT FORGET THAT THE IDASH DATASET IS DIFFERENT FROM THE MNIST DATASET!


# Calculate the 2/(.25*SUM{x[i][j]}) * .9
sumxij = 0.0
for row in X:
	for rowi in row:
		sumxij += rowi
sumxij = sumxij/4.0
x0 = 2.0 / sumxij  *    .9
print 'x0 = 2.0 / sumxij * .9 = ', x0

#random.shuffle(X)
#should shuffle [Y,X] together!
'''
Z = zip(Y,X)
random.shuffle(Z)
#should shuffle [Y,X] together!
X = [item[1] for item in Z]
Y = [item[0] for item in Z]
'''

#     Step 1. Extract data from a csv file
#with open('MNISTtrain3(+1)8(-1)with14x14.csv','r') as csvfile:
with open('MNISTt10k3(+1)8(-1)with14x14.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	#reader.next() # leave behind the first row
	testdata = []
	for row in reader:
		# reader.next() return a string
		row = [float(x) for x in row]
		testdata.append(row)
csvfile.close()
#     Step 4. Extract testX and testY from testdata
testX = [[1]+row[1:] for row in testdata[:]]
for colidx in range(len(testX[0])):
	colmax = 1.0
	for (rowidx, row) in enumerate(testX):
		if row[colidx] > colmax :
			colmax = row[colidx]
	for (rowidx, row) in enumerate(testX):
		row[colidx] /= colmax
testY = [int(row[0]) for row in testdata[:]]
# turn y{+0,+1} to y{-1,+1}
#testY = [2*y-1 for y in testY]    # DONT FORGET THAT THE IDASH DATASET IS DIFFERENT FROM THE MNIST DATASET!


hlambda = lambda x:1.0/(1+exp(-x))
#hlambda = lambda x:5.0000e-01  +1.7786e-01*x  -3.6943e-03*pow(x,3)  +3.6602e-05*pow(x,5)  -1.2344e-07*pow(x,7)


MAX_ITER = 30


'''
-------------------------------------------------------------------------------------------
------------------------- The Presented Method: Adagrad With .25XTXasG --------------------
-------------------------------------------------------------------------------------------
------------------------- Adagrad with G (SFH directly by 0.25XTX) ------------------------
-------------------------------------------------------------------------------------------
''' 

# Stage 2. 
#     Step 1. Initialize Simplified Fixed Hessian Matrix
MX = np.matrix(X)
MXT = MX.T
MXTMX = MXT.dot(MX)                 

# H = +1/4 * X.T * X            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
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
NewtonIter = 9
MB_inv = np.matrix(np.eye(mxtmx.shape[0]))
for idx in range(mxtmx.shape[0]):
	MB_inv[idx,idx] = x0 
for iter in range(NewtonIter):
	for idx in range(mxtmx.shape[0]):
		MB_inv[idx,idx] = MB_inv[idx,idx]*( 2- mb[idx,idx]*MB_inv[idx,idx] )
	print MB_inv
	print '----------------------------------------------------------------------------------'
	print '----------------------------------------------------------------------------------'

# get the inverse of matrix MB in advance


#     Step 2. Initialize Weight Vector (n x 1)
# Setting the initial weight to 1 leads to a large input to sigmoid function,
# which would cause a big problem to this algorithm when using polynomial
# to substitute the sigmoid function. So, it is a good choice to set w = 0.

# [[0]... to make MW a column vector(matrix)
W = [[0.0] for x in range(MB.shape[0])]
MW = np.matrix(W)

#     Step 2. Set the Maximum Iteration and Record each cost function
EmethodAdagradwith_MLE = []
EmethodAdagradwith_AUC = []
EmethodAdagradwith_SIGMOID = []


# Stage 3.
#     Start the Gradient Descent algorithm
#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
#           grad = [Y@(1 - sigm(yWTx))]T * X

Gt = np.matrix(np.eye(mxtmx.shape[0]))
for idx in range(mxtmx.shape[0]):
	Gt[idx,idx] = epsilon*epsilon

for iter in range(MAX_ITER):
	curSigmoidInput = []

#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
	# W.T * X
	MXW = MX * MW
	# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
	yhypothesis = []
	for idx in range(len(Y)):
		#print Y[idx]*MXV.A[idx][0], '\t', 1 - hlambda(Y[idx]*MXV.A[idx][0])
		curSigmoidInput.append(Y[idx]*MXW.A[idx][0])

		# the polynomial to substitute the Sigmoid function
		# y = 0.5 +2.3097e-01*x -1.1156e-02*x^3 +3.1533e-04*x^5 -3.2963e-06*x^7;
		h = 1 - hlambda(Y[idx]*MXW.A[idx][0])

		yhypothesis.append([h*Y[idx]])

	Myhypothesis = np.matrix(yhypothesis)
	# g = [Y@(1 - sigm(yWTx))]T * X	
	Mg = MXT * Myhypothesis
	
	#MG = (.99 + MB_inv) * Mg
	#MG = Mg   

	MG = MB_inv * Mg

	for ixx in range(mxtmx.shape[0]):
		Gt[ixx,ixx] += MG.A[ixx][0] * MG.A[ixx][0]    

	Gamma = np.matrix(np.eye(mxtmx.shape[0]))
	for idx in range(mxtmx.shape[0]):
		Gamma[idx,idx] = (1.0 +.01) / math.sqrt(Gt[idx,idx]) 
		#Gamma[idx,idx] = .01 / math.sqrt(Gt[idx,idx])

	# should be 'plus', 'cause to compute the MLE       
	MW = MW + Gamma* MG
	#MW = MW + Gamma* MG


#     Step 3. Update the Weight Vector using Hessian Matrix and gradient  
	EmethodAdagradwith_SIGMOID.append(curSigmoidInput)         
#     Step 4. Calculate the cost function using Maximum likelihood Estimation
	# log-likelihood
	newMtestXV = MX * MW
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-Y[idx]*newMtestXV.A[idx][0]))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	EmethodAdagradwith_MLE.append(loglikelihood)

	#------------------------------------------------------------------------
	#---- y = { 0, 1} --- WHAT IS THE PROBLEM WITH MLE ? --- y = {-1,+1} ----
	#------------------------------------------------------------------------

	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-newMtestXV.A[idx][0]))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	EmethodAdagradwith_AUC.append(ROCAUC(hxlist, Y))


'''
-------------------------------------------------------------------------------------------
----------------------------------- end of experiments -----------------------------------
-------------------------------------------------------------------------------------------
'''

'''
-------------------------------------------------------------------------------------------
------------------------- The Presented Method: Adagrad -----------------------------------
-------------------------------------------------------------------------------------------
----------------------------------------- Adagrad -----------------------------------------
-------------------------------------------------------------------------------------------
''' 

# Stage 2. 
#     Step 1. Initialize Simplified Fixed Hessian Matrix
MX = np.matrix(X)
MXT = MX.T
MXTMX = MXT.dot(MX)                 

# H = +1/4 * X.T * X            #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
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
NewtonIter = 9
MB_inv = np.matrix(np.eye(mxtmx.shape[0]))
for idx in range(mxtmx.shape[0]):
	MB_inv[idx,idx] = x0 
for iter in range(NewtonIter):
	for idx in range(mxtmx.shape[0]):
		MB_inv[idx,idx] = MB_inv[idx,idx]*( 2- mb[idx,idx]*MB_inv[idx,idx] )
	print MB_inv
	print '----------------------------------------------------------------------------------'
	print '----------------------------------------------------------------------------------'

# get the inverse of matrix MB in advance


#     Step 2. Initialize Weight Vector (n x 1)
# Setting the initial weight to 1 leads to a large input to sigmoid function,
# which would cause a big problem to this algorithm when using polynomial
# to substitute the sigmoid function. So, it is a good choice to set w = 0.

# [[0]... to make MW a column vector(matrix)
W = [[0.0] for x in range(MB.shape[0])]
MW = np.matrix(W)

#     Step 2. Set the Maximum Iteration and Record each cost function
EmethodAdagrad_MLE = []
EmethodAdagrad_AUC = []
EmethodAdagrad_SIGMOID = []


# Stage 3.
#     Start the Gradient Descent algorithm
#     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
#           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
#                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
#           grad = [Y@(1 - sigm(yWTx))]T * X

Gt = np.matrix(np.eye(mxtmx.shape[0]))
for idx in range(mxtmx.shape[0]):
	Gt[idx,idx] = epsilon

for iter in range(MAX_ITER):
	curSigmoidInput = []

#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
	# W.T * X
	MXW = MX * MW
	# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
	yhypothesis = []
	for idx in range(len(Y)):
		#print Y[idx]*MXV.A[idx][0], '\t', 1 - hlambda(Y[idx]*MXV.A[idx][0])
		curSigmoidInput.append(Y[idx]*MXW.A[idx][0])

		# the polynomial to substitute the Sigmoid function
		# y = 0.5 +2.3097e-01*x -1.1156e-02*x^3 +3.1533e-04*x^5 -3.2963e-06*x^7;
		h = 1 - hlambda(Y[idx]*MXW.A[idx][0])

		yhypothesis.append([h*Y[idx]])

	Myhypothesis = np.matrix(yhypothesis)
	# g = [Y@(1 - sigm(yWTx))]T * X	
	Mg = MXT * Myhypothesis
	
	#MG = (.99 + MB_inv) * Mg
	#MG = Mg   
	#MG = MB_inv * Mg

	for ixx in range(mxtmx.shape[0]):
		Gt[ixx,ixx] += Mg.A[ixx][0] * Mg.A[ixx][0]    

	Gamma = np.matrix(np.eye(mxtmx.shape[0]))
	for idx in range(mxtmx.shape[0]):
		Gamma[idx,idx] = .01 / math.sqrt(Gt[idx,idx]) 
		#Gamma[idx,idx] = .01 / math.sqrt(Gt[idx,idx])

	# should be 'plus', 'cause to compute the MLE       
	MW = MW + Gamma* Mg
	#MW = MW + Gamma* MG


#     Step 3. Update the Weight Vector using Hessian Matrix and gradient  
	EmethodAdagrad_SIGMOID.append(curSigmoidInput)         
#     Step 4. Calculate the cost function using Maximum likelihood Estimation
	# log-likelihood
	newMtestXV = MX * MW
	loghx = []
	for idx in range(len(Y)):
		# WARNING: iff y in {-1,+1}
		loghxi = -log(1+exp(-Y[idx]*newMtestXV.A[idx][0]))
		loghx.append(loghxi)
	loglikelihood = sum(loghx)
	EmethodAdagrad_MLE.append(loglikelihood)

	#------------------------------------------------------------------------
	#---- y = { 0, 1} --- WHAT IS THE PROBLEM WITH MLE ? --- y = {-1,+1} ----
	#------------------------------------------------------------------------

	# BE CAREFULL WITH INPUT LABELS!
	newhypothesis = []
	for idx in range(len(Y)):
		hx = 1.0/(1+exp(-newMtestXV.A[idx][0]))
		newhypothesis.append(hx)
	hxlist = [ hx for hx in newhypothesis ]
	print iter, '-th AUC : ', ROCAUC(hxlist, Y), ' MLE = ', loglikelihood
	EmethodAdagrad_AUC.append(ROCAUC(hxlist, Y))


'''
-------------------------------------------------------------------------------------------
----------------------------------- end of experiments -----------------------------------
-------------------------------------------------------------------------------------------
'''
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.ylabel(r'\textit{voltage} (mV)',fontsize=16)
label = [ 'Adagrad + G', 'Adagrad' ]
#EmethodAdagradwith_MLE[0] = None; EmethodAdagradwith_MLE[1] = None;
#EmethodAdagrad_MLE[0] = None;     EmethodAdagrad_MLE[1] = None;
print EmethodAdagrad_MLE
EmethodAdagradwith_MLE = [-math.log(-ele, 10) for ele in EmethodAdagradwith_MLE]
EmethodAdagrad_MLE = [-math.log(-ele, 10) for ele in EmethodAdagrad_MLE]
print EmethodAdagrad_MLE
#EmethodAdagradwith_MLE = [None] + EmethodAdagradwith_MLE[:]
#EmethodAdagrad_MLE = [None] + EmethodAdagrad_MLE[:]

plt.plot(range(len(EmethodAdagradwith_MLE)), EmethodAdagradwith_MLE, 'v-b')
plt.plot(range(len(EmethodAdagrad_MLE)), EmethodAdagrad_MLE, 'v-r')
#plt.axis("equal")
#plt.title('MLE Score on MNIST Training Dataset')
plt.xlabel('Iteration Number')
plt.ylabel("Maximum Log-likelihood Estimation: -log(-MLE)")
#plt.xlim([1, len(EmethodAdagrad_MLE)])
plt.legend(label, loc = 4, ncol = 1)   
plt.grid()
plt.show()
#AUC Score on split11 Dataset
#plt.savefig("MLE4.png")
#plt.close()

plt.plot(range(len(EmethodAdagradwith_AUC)), EmethodAdagradwith_AUC, 'v-b')
plt.plot(range(len(EmethodAdagrad_AUC)), EmethodAdagrad_AUC, 'v-r')
#plt.title('AUC Score on MNIST Training Dataset')
plt.xlabel("Iteration Number")
plt.ylabel("Area Under the Curve")
#plt.xlim([1, len(EmethodAdagrad_AUC)])
plt.legend(label, loc = 4, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("AUC4.png")
#plt.close()

#plt.close()
for iter in range(len(EmethodAdagradwith_SIGMOID)):
	plt.plot([iter]*len(EmethodAdagradwith_SIGMOID[iter]), EmethodAdagradwith_SIGMOID[iter], 'v-b')
	miny = min(EmethodAdagradwith_SIGMOID[iter])
	maxy = max(EmethodAdagradwith_SIGMOID[iter])
	plt.text(iter, miny, '%.3f' % miny, ha='center', va='top', fontsize=10)
	plt.text(iter, maxy, '%.3f' % maxy, ha='center', va='bottom', fontsize=10) 
	print '[ ',min(EmethodAdagradwith_SIGMOID[iter]), ' , ', max(EmethodAdagradwith_SIGMOID[iter]), ' ]  '
plt.title('INPUT RANGE OF SIGMOID : Nesterov + .25XTXasG')
plt.grid()
plt.show()




# -------------- FILE: MLE -------------- 
# -- Iterations -- Adagrad -- AdagradG -- 
filePath = 'ICANN2020_PythonExperiment_MNIST_Adagradvs.AdagradG_MLE(-log(-x)).csv';
PythonExperimentMNIST =      open(filePath,      'w')
PythonExperimentMNIST =      open(filePath,      'a+b')

PythonExperimentMNIST.write('Iterations'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('Adagrad');    
PythonExperimentMNIST.write(','); 
PythonExperimentMNIST.write('AdagradG');   
PythonExperimentMNIST.write("\n");

for (idx, ele) in enumerate(EmethodAdagradwith_MLE):
	PythonExperimentMNIST.write(str(idx)); 
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(EmethodAdagrad_MLE[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(EmethodAdagradwith_MLE[idx]));
	PythonExperimentMNIST.write("\n");

PythonExperimentMNIST.close();

# -------------- FILE: MLE -------------- 
# -- Iterations -- Adagrad -- AdagradG -- 
filePath = 'ICANN2020_PythonExperiment_MNIST_Adagradvs.AdagradG_AUC.csv';
PythonExperimentMNIST =      open(filePath,      'w')
PythonExperimentMNIST =      open(filePath,      'a+b')

PythonExperimentMNIST.write('Iterations'); 
PythonExperimentMNIST.write(',');
PythonExperimentMNIST.write('Adagrad');    
PythonExperimentMNIST.write(','); 
PythonExperimentMNIST.write('AdagradG');   
PythonExperimentMNIST.write("\n");

for (idx, ele) in enumerate(EmethodAdagradwith_AUC):
	PythonExperimentMNIST.write(str(idx)); 
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(EmethodAdagrad_AUC[idx]));
	PythonExperimentMNIST.write(',');
	PythonExperimentMNIST.write(str(EmethodAdagradwith_AUC[idx]));
	PythonExperimentMNIST.write("\n");

PythonExperimentMNIST.close();