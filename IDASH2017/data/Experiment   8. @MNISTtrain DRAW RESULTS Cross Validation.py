# coding=utf8
# 2019-04-14 07:16 a.m. GMT +08：00
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

import csv

with open('Experiment 8 IDASH2017 Bonte MLE.csv','rb') as csvfile:
	reader = csv.reader(csvfile)
	data = []
	for row in reader:
		# reader.next() return a string
		row = [float(x) for x in row]
		data.append(row)
csvfile.close()
EBonte = [0.0 for x in range(len(data[0]))]
for row in data:
	for idx in range(len(data[0])):
		EBonte[idx] += row[idx]
for idx in range(len(data[0])):
	EBonte[idx] /= len(data)

with open('Experiment 8 IDASH2017 Bonte with rate MLE.csv','rb') as csvfile:
	reader = csv.reader(csvfile)
	reader.next() # leave behind the first row
	data = []
	for row in reader:
		# reader.next() return a string
		row = [float(x) for x in row]
		data.append(row)
csvfile.close()
EBontewithrate = [0.0 for x in range(len(data[0]))]
for row in data:
	for idx in range(len(data[0])):
		EBontewithrate[idx] += row[idx]
for idx in range(len(data[0])):
	EBontewithrate[idx] /= len(data)

with open('Experiment 8 IDASH2017 Nesterov MLE.csv','rb') as csvfile:
	reader = csv.reader(csvfile)
	reader.next() # leave behind the first row
	data = []
	for row in reader:
		# reader.next() return a string
		row = [float(x) for x in row]
		data.append(row)
csvfile.close()
ENesterov = [0.0 for x in range(len(data[0]))]
for row in data:
	for idx in range(len(data[0])):
		ENesterov[idx] += row[idx]
for idx in range(len(data[0])):
	ENesterov[idx] /= len(data)


with open('Experiment 8 IDASH2017 Nesterov with G MLE.csv','rb') as csvfile:
	reader = csv.reader(csvfile)
	reader.next() # leave behind the first row
	data = []
	for row in reader:
		# reader.next() return a string
		row = [float(x) for x in row]
		data.append(row)
csvfile.close()
ENesterovwithG = [0.0 for x in range(len(data[0]))]
for row in data:
	for idx in range(len(data[0])):
		ENesterovwithG[idx] += row[idx]
for idx in range(len(data[0])):
	ENesterovwithG[idx] /= len(data)

'''
with open('Experiment 8 IDASH2017 Sin(x) AUC.csv','rb') as csvfile:
	reader = csv.reader(csvfile)
	reader.next() # leave behind the first row
	data = []
	for row in reader:
		# reader.next() return a string
		row = [float(x) for x in row]
		data.append(row)
csvfile.close()
Esin = [0.0 for x in range(len(data[0]))]
for row in data:
	for idx in range(len(data[0])):
		Esin[idx] += row[idx]
for idx in range(len(data[0])):
	Esin[idx] /= len(data)
'''

label = ["Bonte's SFH",'Nesterov', 'Method1 (SFH with rate)', 'Method4 (Nesterov +G by .25XTX)','sin as h(x)']
label = ["Bonte's SFH",'Nesterov', 'Method1 (SFH with rate)', 'Method4 (Nesterov +G by .25XTX)']
#label = ["Bonte's SFH", 'Method1 (SFH with rate)', 'Method4 (Nesterov +G by .25XTX)']
plt.plot(range(len(EBonte)),   EBonte,     's-k')
plt.plot(range(len(ENesterov)), ENesterov, 'v-b')
plt.plot(range(len(EBontewithrate)), EBontewithrate,   '>-m')
plt.plot(range(len(ENesterovwithG)), ENesterovwithG,   'p-g')
#plt.plot(range(len(Esin)), Esin,   '>-m')
#plt.axis("equal")
plt.title('MLE')
plt.legend(label, loc = 0, ncol = 1)  
plt.grid()
plt.show()
#plt.savefig("MLE4.png")
#plt.close()