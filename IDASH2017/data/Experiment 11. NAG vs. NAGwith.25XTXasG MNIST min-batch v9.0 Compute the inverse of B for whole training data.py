# coding=utf8
# 2019-12-04 09:43 a.m. GMT +08：00
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
# The ROC curve is constructed by plotting the (X=FPR, Y=TPR) pairs for each possible value
# of the threshold. By computing the TPR and FPR for varying thresholds in [0,1], we can construct
# the receiver operating characteristic curve or ROC-curve.
#     +--------------------------------------- actual class --------------------------+
#     +-----------------------              -1               +1 ----------------------+
#     +------------- predicted  -1   true negative (TN) false negative (FN) ----------+
#     +-------------     class  +1   false positive(FP) true positive (TP)  ----------+
#     +-------------------------------------------------------------------------------+
#     +------------- X = FPR = #FP/(#FP + #TN) ---- Y = TPR = #TP/(#TP + #FN) --------+
#     +-------------------------------------------------------------------------------+
#            X:实际负类中预测结果(正类)出错的比例       Y:实际正类中预测结果(正类)的比例
# -----------------------------------------------------------------------------------------
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

def ACC(score, y, show=False):
	POSITIVE = y.count(+1)
	NEGATIVE = y.count(-1)            # WARNNING: WHAT ARE THE LABELS ?

	TP = 0
	TN = 0
	# need (score,y) to be sorted
	for (i, s) in enumerate(score):
		if y[i]==+1 and s>=0.5:
			TP = TP + 1
		if y[i]==-1 and s <0.5:                  # WARNNING: WHAT ARE THE LABELS ?
			TN = TN + 1

	#print 'TP = ',TP, ', TN = ',TN,
	ACC = float(TN + TP)/len(y)
	return ACC

print '----------------------------------------------------------------------------------'
print "------------- Experiment11. Nesterov With .25XTXasG                  -------------"
print '-------------    Data Set : MNIST                                    -------------'
print '-------------           X : [[1,x11,x12,...],[1,x21,x22,...],...]    -------------'
print '-------------           Y : y = {-1, +1}                             -------------'
print '----------------------------------------------------------------------------------'

import csv

# Stage 1. 
#     Step 1. Extract test data from a csv file
#with open('MNISTt10k3(+1)8(-1)with14x14.csv','r') as csvfile:
with open('MNISTtrain3(+1)8(-1)with14x14.csv','r') as csvfile:
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

#     Step 1. Extract train data from a csv file
with open('MNISTtrain3(+1)8(-1)with14x14.csv','r') as csvfile:
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

#X = X[:1000]
#Y = Y[:1000]

# Calculate the 2/(.25*SUM{x[i][j]}) * .9 for Newton Method to converge
sumxij = 0.0
for row in X:
	for rowi in row:
		sumxij += rowi
sumxij = sumxij/4.0
x0 = 2.0 / sumxij  *    .9
print 'x0 = 2.0 / sumxij * .9 = ', x0
# Compute the start for Newton method to invert the diagonal element of B

batchsize = 1024
Xminbatches = []
idx=0
while True:
	if idx+batchsize > len(X): break
	batch=[]
	for i in range(idx,idx+batchsize):
		batch.append(X[i])
	Xminbatches.append(batch)
	idx += batchsize
if len(X)%batchsize != 0:
	batch=[]
	for i in range(idx, len(X)):
		batch.append(X[i])
	Xminbatches.append(batch)	
print 'len(X) = ', len(X)
print 'batchsize = ', batchsize
print 'len(X)%batchsize = ',len(X)%batchsize
#print 'Xminbatches[0]', Xminbatches[0]
#print 'Xminbatches[-1]', Xminbatches[-1]
#print 'Xminbatches'
Yminbatches = []
idx=0
while True:
	if idx+batchsize > len(Y): break
	batch=[]
	for i in range(idx,idx+batchsize):
		batch.append(Y[i])
	Yminbatches.append(batch)
	idx += batchsize
if len(Y)%batchsize != 0:
	batch=[]
	for i in range(idx, len(Y)):
		batch.append(Y[i])
	Yminbatches.append(batch)	
print 'Yminbatches[0]', Yminbatches[0]
print 'Yminbatches[-1]', Yminbatches[-1]
print 'len(Yminbatches[-1]) = ', len(Yminbatches[-1])

#should shuffle [Y,X] together in the Iteration Processes! 
'''
Z = zip(Y,X)
random.shuffle(Z)
#should shuffle [Y,X] together!
X = [item[1] for item in Z]
Y = [item[0] for item in Z]
'''


hlambda = lambda x:1.0/(1+exp(-x))
#hlambda = lambda x:5.0000e-01  +1.7786e-01*x  -3.6943e-03*pow(x,3)  +3.6602e-05*pow(x,5)  -1.2344e-07*pow(x,7)


MAX_ITER = 30

# used for the input range of sigmoid
def limit(inputx):
	#return inputx
	if inputx > 25 : 
		return 25
	if inputx < -25 :
		return -25;
	return inputx

'''
-------------------------------------------------------------------------------------------
-------------- The Presented Method: Nesterov Accelerated Gradient for min-batch ----------
-------------------------------------------------------------------------------------------
------------------------------ Nesterov Accelerated Gradient ------------------------------
-------------------------------------------------------------------------------------------
''' 
# Stage 2. 
#     Step 1. Initialize Simplified Fixed Hessian Matrix For each min-batch                
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
	mb[idx,idx] = (+.25)*mxtmx[idx, 0]  + epsilon           
	print 'M[',idx,'][',idx,'] = ',mb[idx,idx]
# END  : Bonte's Specific Order On XTX

MB = mb    
print MB 
# Use Newton Method to calculate the inverses of MB[i][i] 
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
V = [[0.0] for x in range(MB.shape[0])]
W = [[0.0] for x in range(MB.shape[0])]
MV = np.matrix(V)
MW = np.matrix(W)

#     Step 2. Set the Maximum Iteration and Record each cost function
Emethod_NAG_MLE = []
Emethod_NAG_AUC = []
Emethod_NAG_ACC = []
Emethod_NAG_SIGMOID = []

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

	#eta = (1 - alpha0) / alpha1
	#gamma = 1.0/(iter+1)/MX.shape[0]
	#gamma = 1.0/(iter+1)/len(Xminbatches[0])
	for (i,Xbatch) in enumerate(Xminbatches):
		eta = (1 - alpha0) / alpha1
		#gamma = 1.0/(iter+1)/len(Xminbatches[0])
		gamma = 1.0/(1 + iter + i)/len(Xminbatches[0])
		#gamma = 1.0/(1 + iter*batchsize + i)

		Ybatch = Yminbatches[i]
		#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
		# W.T * X
		MXbatch = np.matrix(Xbatch)
		MXbatchV = MXbatch * MV
		# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
		yhypothesis = []
		for idx in range(len(Ybatch)):

			curSigmoidInput.append(Ybatch[idx]*MXbatchV.A[idx][0])

			# hlambda(): the polynomial function to substitute the Sigmoid function
			h = 1 - hlambda(limit(Ybatch[idx]*MXbatchV.A[idx][0]))

			yhypothesis.append([h*Ybatch[idx]])

		Myhypothesis = np.matrix(yhypothesis)
		# g = [Y@(1 - sigm(yWTx))]T * X	
		# maybe should put the 1/m only before the alpha
		#Mg = 1.0/len(Xbatch) * MXbatch.T * Myhypothesis 
		Mg = MXbatch.T * Myhypothesis  

		
		#MG = MB_inv * Mg          
		# should be 'plus', 'cause to compute the MLE  
		#MtmpW = MV + (1.0 + gamma) * MG   
		MtmpW = MV + gamma * Mg       
		MV = (1.0-eta)*MtmpW + (eta)*MW
		MW = MtmpW


		Emethod_NAG_SIGMOID.append(curSigmoidInput)         
	#     Step 4. Calculate the cost function using Maximum likelihood Estimation
		# log-likelihood
		MtestX = np.matrix(testX)
		newMtestXV = MtestX * MV
		loghx = []
		for idx in range(len(testY)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(limit(-testY[idx]*newMtestXV.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_NAG_MLE.append(loglikelihood)


		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(testY)):
			hx = 1.0/(1+exp(limit(-newMtestXV.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		print iter, '-th AUC : ', ROCAUC(hxlist, testY), ' MLE = ', loglikelihood, ' ACC = ', ACC(hxlist, testY)
		Emethod_NAG_AUC.append(ROCAUC(hxlist, testY))
		Emethod_NAG_ACC.append(ACC(hxlist, testY))

		alpha0 = alpha1
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0


'''
-------------------------------------------------------------------------------------------
----------------------------------- end of experiments -----------------------------------
-------------------------------------------------------------------------------------------
'''

'''
-------------------------------------------------------------------------------------------
------------------ The Presented Method: Nesterov + .25XTXasG for min-batch ---------------
-------------------------------------------------------------------------------------------
------------------------ Nesterov with G (SFH directly by 0.25XTX) ------------------------
-------------------------------------------------------------------------------------------
''' 

# Stage 2. 
#     Step 1. Initialize Simplified Fixed Hessian Matrix For each min-batch                
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
	mb[idx,idx] = (+.25)*mxtmx[idx, 0]  + epsilon           
	print 'M[',idx,'][',idx,'] = ',mb[idx,idx]
# END  : Bonte's Specific Order On XTX

MB = mb    
print MB 
# Use Newton Method to calculate the inverses of MB[i][i] 
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
V = [[0.0] for x in range(MB.shape[0])]
W = [[0.0] for x in range(MB.shape[0])]
MV = np.matrix(V)
MW = np.matrix(W)

#     Step 2. Set the Maximum Iteration and Record each cost function
Emethod_NAGG_MLE = []
Emethod_NAGG_AUC = []
Emethod_NAGG_ACC = []
Emethod_NAGG_SIGMOID = []

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

	#eta = (1 - alpha0) / alpha1
	#gamma = 1.0/(iter+1)/len(Xminbatches[0])
	for (i,Xbatch) in enumerate(Xminbatches):
		eta = (1 - alpha0) / alpha1
		#gamma = 1.0/(iter+1)/len(Xminbatches[0])
		gamma = 1.0/(1 + iter + i)/len(Xminbatches[0])
		#gamma = 1.0/(1 + iter*batchsize + i)

		Ybatch = Yminbatches[i]
		#     Step 1. Calculate the Gradient = [Y@(1 - sigm(Y@WT*X))]T * X
		# W.T * X
		MXbatch = np.matrix(Xbatch)
		MXbatchV = MXbatch * MV
		# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)
		yhypothesis = []
		for idx in range(len(Ybatch)):

			curSigmoidInput.append(Ybatch[idx]*MXbatchV.A[idx][0])

			# hlambda(): the polynomial function to substitute the Sigmoid function
			h = 1 - hlambda(limit(Ybatch[idx]*MXbatchV.A[idx][0]))

			yhypothesis.append([h*Ybatch[idx]])

		Myhypothesis = np.matrix(yhypothesis)
		# g = [Y@(1 - sigm(yWTx))]T * X	
		# maybe should put the 1/m only before the alpha
		#Mg = 1.0/len(Xbatch) * MXbatch.T * Myhypothesis 
		Mg = MXbatch.T * Myhypothesis  

		MX = np.matrix(Xbatch)
		MXT = MX.T
		MXTMX = MXT.dot(MX)  
		epsilon = 1e-08
		mx = MX.sum(axis=1)
		mxtmx = MX.T.dot(mx)
		mb = np.matrix(np.eye(mxtmx.shape[0]))
		for idx in range(mxtmx.shape[0]):
			mb[idx,idx] = (+.25)*mxtmx[idx, 0]  + epsilon           
		MB_inv = mb
		for idx in range(mxtmx.shape[0]):
			MB_inv[idx,idx] = 1.0/mb[idx,idx] 


		MG = MB_inv * Mg          
		# should be 'plus', 'cause to compute the MLE  
		MtmpW = MV + (1.0 + gamma) * MG   
		#MtmpW = MV + gamma * Mg       
		MV = (1.0-eta)*MtmpW + (eta)*MW
		MW = MtmpW


		Emethod_NAGG_SIGMOID.append(curSigmoidInput)         
	#     Step 4. Calculate the cost function using Maximum likelihood Estimation
		# log-likelihood
		MtestX = np.matrix(testX)
		newMtestXV = MtestX * MV
		loghx = []
		for idx in range(len(testY)):
			# WARNING: iff y in {-1,+1}
			loghxi = -log(1+exp(limit(-testY[idx]*newMtestXV.A[idx][0])))
			loghx.append(loghxi)
		loglikelihood = sum(loghx)
		Emethod_NAGG_MLE.append(loglikelihood)


		# BE CAREFULL WITH INPUT LABELS!
		newhypothesis = []
		for idx in range(len(testY)):
			hx = 1.0/(1+exp(limit(-newMtestXV.A[idx][0])))
			newhypothesis.append(hx)
		hxlist = [ hx for hx in newhypothesis ]
		print iter, '-th AUC : ', ROCAUC(hxlist, testY), ' MLE = ', loglikelihood, ' ACC = ', ACC(hxlist, testY)
		Emethod_NAGG_AUC.append(ROCAUC(hxlist, testY))
		Emethod_NAGG_ACC.append(ACC(hxlist, testY))

		alpha0 = alpha1
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0


'''
-------------------------------------------------------------------------------------------
----------------------------------- end of experiments -----------------------------------
-------------------------------------------------------------------------------------------
'''


label = [ 'NAG', 'NAG + G' ]
plt.plot(range(len(Emethod_NAG_MLE)), Emethod_NAG_MLE, 'v-b')
plt.plot(range(len(Emethod_NAGG_MLE)), Emethod_NAGG_MLE, '^--r')
#plt.axis("equal")
plt.title('MLE')
plt.legend(label, loc = 0, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("MLE4.png")
#plt.close()

plt.plot(range(len(Emethod_NAG_AUC)), Emethod_NAG_AUC, 'v-b')
plt.plot(range(len(Emethod_NAGG_AUC)), Emethod_NAGG_AUC, '^--r')
plt.title('AUC')
plt.legend(label, loc = 0, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("AUC4.png")
#plt.close()

plt.plot(range(len(Emethod_NAG_ACC)), Emethod_NAG_ACC, 'v-b')
plt.plot(range(len(Emethod_NAGG_ACC)), Emethod_NAGG_ACC, '^--r')
plt.title('ACC')
plt.legend(label, loc = 0, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("AUC4.png")
#plt.close()

'''
plt.close()
for iter in range(len(Emethod_SIGMOID)):
	plt.plot([iter]*len(Emethod_SIGMOID[iter]), Emethod_SIGMOID[iter], 'v-b')
	miny = min(Emethod_SIGMOID[iter])
	maxy = max(Emethod_SIGMOID[iter])
	plt.text(iter, miny, '%.3f' % miny, ha='center', va='top', fontsize=10)
	plt.text(iter, maxy, '%.3f' % maxy, ha='center', va='bottom', fontsize=10) 
	print '[ ',min(Emethod_SIGMOID[iter]), ' , ', max(Emethod_SIGMOID[iter]), ' ]  '
plt.title('INPUT RANGE OF SIGMOID : Nesterov + .25XTXasG')
plt.grid()
plt.show()
'''

