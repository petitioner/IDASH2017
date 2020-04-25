/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#ifndef IDASH2017_MYMETHODS_H_
#define IDASH2017_MYMETHODS_H_
#include "MyTools.h"
#include "Scheme.h"

class MyMethods {
public:

	static void    testPlainFiveFoldCrossValidation(double** data, double* label, long factorDim, long sampleDim, long numIter, long NUMfold);
	static void    testPlainTrainingAndTesting1time(double** traindata, double* trainlabel, long trainSampleDim, long factorDim, double** testdata, double* testlabel, long testSampleDim, long numIter, long NUMfold);
	static void    testPlainTrainingAndTesting5time(double** traindata, double* trainlabel, long trainSampleDim, long factorDim, double** testdata, double* testlabel, long testSampleDim, long numIter, long NUMfold);
	static void    testPlainSpeedofConvergence5time(double** data, double* label, long factorDim, long sampleDim, long numIter, long NUMfold);
	static double* testPlainNesterov(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath);
	static double* testPlainNesterovWithG(double** traindata, double* trainlabel, long factorDim, long trainSampleDim,	long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath);
	static double* testPlainBonteSFH(double** traindata, double* trainlabel, long factorDim, long trainSampleDim,	long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath);
	static double* testPlainBonteSFHwithLearningRate(double** traindata, double* trainlabel, long factorDim, long trainSampleDim,	long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath);


	static long suggestLogN(long lambda, long logQ);

	static void testEncNLGD(double** zDataTrain, double** zDataTest, long factorDim, long sampleDimTrain, long sampleDimTest,
			bool isYfirst, long numIter, long k, double gammaUp, double gammaDown, bool isInitZero);

	static void testEncNLGDFOLD(long fold, double** zData, long factorDim, long sampleDim,
			bool isYfirst, long numIter, long k, double gammaUp, double gammaDown, bool isInitZero);

	static void testPlainNLGD(double** zDataTrain, double** zDataTest, long factorDim, long sampleDimTrain, long sampleDimTest,
			bool isYfirst, long numIter, long k, double gammaUp, double gammaDown, bool isInitZero);

	//static void bootstrap(Ciphertext& cipher, long logq, long logQ, long logT, long logI = 4);
	static void bootstrap(Scheme& scheme, Ciphertext* &encVData, Ciphertext* &encWData, long cnum, long slots, long trainSampleDim, long batch, long logQ, long logT=3, long logI=4);
	static void bootstrap(Scheme& scheme, Ciphertext* &encVData, long cnum, long slots, long trainSampleDim, long batch, long logQ, long logT=3, long logI=4);


	static double* testCryptoBonteSFHwithLearningRate(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath);

	static double* testCryptoNesterovWithG(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath);
};

#endif /* MYMETHODS_H_ */
