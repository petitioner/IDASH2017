/*
 * Copyright (c) by CryptoLab inc.
 * This program is licensed under a
 * Creative Commons Attribution-NonCommercial 3.0 Unported License.
 * You should have received a copy of the license along with this
 * work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
 */

#include <NTL/BasicThreadPool.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>

#include "MyMethods.h"
#include "MyTools.h"

using namespace std;
using namespace NTL;

/*
 * run: ./IDASH2017 trainfile isYfirst numIter k gammaUp gammaDown isInitZero fold isEncrypted testfile
 * ./HEML string bool long long double double bool long bool string
 * example: ./IDASH2017 "../data/data103x1579.txt" 1 7 5 1 -1 1 5 1
 * example: ./IDASH2017 "../data/1_training_data_csv" 1 7 5 1 -1 1 0 1 "../data/1_testing_data_csv"
 *
 * parameters:
 * trainfile - path to train file
 * isYfirst - {0,1} y parameter first OR last
 * numIter - number of iterations
 * kdeg - degree of sigmoid approximation function k in {3,5,7}
 * gammaUp - corresponds to learning rate
 * gammaDown - corresponds to learning rate
 * isInitZero - is initial weights zero or average
 * fold - folding method if arguments <= 8 we use folding method
 * isEncrypted - encrypted or plain
 * testfile - path to test file (checks if number of arguments > 8 then we use standard method
 *
 * current files that in data folder (filename isYfirst):
 * "../data/data103x1579.txt" true
 *
 */

int main(int argc, char **argv) {
	// example: ./IDASH2017 "../data/data103x1579.txt" 1 7 5 1 -1 1 5 1
	//	size_t currentAfterSchemeSize = getCurrentRSS( ) >> 20;
	//	size_t peakAfterSchemeSize = getPeakRSS() >> 20;
	//	cout << "Current Memory Usage After Scheme Generation: " << currentAfterSchemeSize << "MB"<< endl;
	//	cout << "Peak Memory Usage After Scheme Generation: " << peakAfterSchemeSize << "MB"<< endl;
	SetNumThreads(8);

	long numIter = 30;
	//long fold = 5;

	/********************************************************************************************************************\
	 ------------------------------------------         THE START LINE         ------------------------------------------
	 --------------------------------------------------------------------------------------------------------------------
	 ---------------------------  Bonte's Simplified Fixed Hessian Newton with Learning Rate  ---------------------------
	 --------------------------------------------------------------------------------------------------------------------
	 ---------------------------   Nesterov’s Accelerated Gradient Descent with .25XTX as G   ---------------------------
	 --------------------------------------------------------------------------------------------------------------------
	 \********************************************************************************************************************/
	cout << "THE START LINE" << endl;
    cout << "FIRST OF ALL, THE PARAMETERS OF POLYNOMIALS FITING THE SIGMOID FUNCTION SHOULD BE REPLACED WITH ANOTHER ONE" << endl << endl;
	/********************************************************************************************************************\
	 ----------------------------  To Test Five Cross Validation In Cipher-Text (IDASH2017)  ----------------------------
	\********************************************************************************************************************/
	cout << " Five-Fold Cross Validation In Cipher-text HAS BEGUN ! " << endl << endl;
	// It would not work in Ubuntu If trainfile = "../data/data103x1579.txt" !
	string trainfile = "../data/data103x1579.txt";
	string testfile = "../data/data103x1579.txt";
	// factorDim includes the class label Y and the feature number of each sample X.
	// sampleDim is the only sample number of the data, excluding the first line for explanation.
	long sampleDim = 0, factorDim = 0;
	double **dataset, **testdataset;
	double *datalabel, *testdatalabel;
	// after called GD::zDataFromFile, factorDim and sampleDim would get the correct value.
	// GD::zDataFromFile will turn the class label y{-0,+1} to y{-1,+1}.
	// GD::zDataFromFile will compute y@X to X, since y and X would always appear together thereafter.
	// if y = -0, then: y = -1 and each X[i] = -1*X[i];  if y = +1, then: y = +1 and each X[i] = +1*X[i];
	double **zData = MyTools::dataFromFile(trainfile, factorDim, sampleDim,	dataset, datalabel);
	double **zDate = MyTools::dataFromFile(testfile, factorDim, sampleDim,	testdataset, testdatalabel);
	// normalize X
	MyTools::normalizeZData(dataset, factorDim, sampleDim);
	MyTools::normalizeZData(testdataset, factorDim, sampleDim);
	// random the order of each row in zData.  "This will dramatically drop down the AUC value. don't know why."?
	//MyTools::shuffleDataSync(dataset, factorDim, sampleDim, datalabel);

	//---------------------------- BonteSFHwithLearningRate ---------------------------
	string pathBonteSFHwithLearningRate = "../result/ExperimentResultInCiphertext_";
	       pathBonteSFHwithLearningRate.append("IDASH2017_");
	       pathBonteSFHwithLearningRate.append("BonteSFHwithXTXasG_");
	string pathNesterovAGwithXTXasG = "../result/ExperimentResultInCiphertext_";
	       pathNesterovAGwithXTXasG.append("IDASH2017_");
	       pathNesterovAGwithXTXasG.append("NesterovwithXTXasG_");

	// Step 1. clear the former result data stored in the several different files.
	// SHOULD BE IN CONSSISTENT WITH THE FILE PATH IN THE EACH ALGORITHM ! eg. "TrainAUC.csv"...
	std::ofstream ofs;
	//ofs.open(pathBonteSFHwithLearningRate + "TrainAUC.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
	//ofs.open(pathBonteSFHwithLearningRate + "TrainMLE.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
	//ofs.open(pathBonteSFHwithLearningRate + "TestAUC.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
	//ofs.open(pathBonteSFHwithLearningRate + "TestMLE.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
	//ofs.open(pathBonteSFHwithLearningRate + "TIME.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
	//ofs.open(pathBonteSFHwithLearningRate + "TIMELabel.csv",std::ofstream::out | std::ofstream::trunc);	ofs.close();
	//ofs.open(pathBonteSFHwithLearningRate + "CurrMEM.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
	//ofs.open(pathBonteSFHwithLearningRate + "PeakMEM.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();

	ofs.open(pathNesterovAGwithXTXasG + "TrainAUC.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
	ofs.open(pathNesterovAGwithXTXasG + "TrainMLE.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
	ofs.open(pathNesterovAGwithXTXasG + "TestAUC.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
	ofs.open(pathNesterovAGwithXTXasG + "TestMLE.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
	ofs.open(pathNesterovAGwithXTXasG + "TIME.csv",			std::ofstream::out | std::ofstream::trunc);	ofs.close();
	ofs.open(pathNesterovAGwithXTXasG + "TIMELabel.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
	ofs.open(pathNesterovAGwithXTXasG + "CurrMEM.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
	ofs.open(pathNesterovAGwithXTXasG + "PeakMEM.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
	// Step 2. If there is a fold cross validate, then cross the testdata

	double **traindata, **testdata;
	double *trainlabel, *testlabel;

	traindata = new double*[sampleDim];
	trainlabel = new double[sampleDim];

	testdata = testdataset;
	testlabel = testdatalabel;

		// use the whole gene dataset (1579 samples) as the train dataset, don't need testdata
		MyTools::shuffleDataSync(dataset, factorDim, sampleDim, datalabel);
		for (long i = 0; i < sampleDim; ++i) {
			traindata[i] = dataset[i];
			trainlabel[i] = datalabel[i];
		}

		//cout << "!!! Crypto: Bonte's Simple Fixed Hessian Newton Method with Learning Rate ( 0.25XTX ) !!!"	<< endl;
		//MyMethods::testCryptoBonteSFHwithLearningRate(traindata, trainlabel, factorDim,
		//		sampleDim, numIter, testdata, testlabel, sampleDim,	pathBonteSFHwithLearningRate);
		//cout << "### Crypto: Bonte's Simple Fixed Hessian Newton Method with Learning Rate ( 0.25XTX ) ###"	<< endl;

		cout << "!!! Crypto: Nesterov's Accelerated Gradient Descent with 0.25XTX as Quadratic Gradient !!!" << endl;
		MyMethods::testCryptoNesterovWithG(traindata, trainlabel, factorDim,
				sampleDim, numIter, testdata, testlabel, sampleDim,	pathNesterovAGwithXTXasG);
		cout << "### Crypto: Nesterov's Accelerated Gradient Descent with 0.25XTX as Quadratic Gradient ###" << endl;


	cout << endl << "END OF THE PROGRAMM" << endl;
	return 0;


//	cout << "FIRST OF ALL, THE PARAMETERS OF POLYNOMIALS FITING THE SIGMOID FUNCTION SHOULD BE REPLACED WITH ANOTHER ONE" << endl << endl;
//
//	/********************************************************************************************************************\
//		 ------------------------  To Test Five Training and Testing In Cipher-Text (MNIST.3[+1]8[-1]) ----------------------
//	 \********************************************************************************************************************/
//
//	string trainfile = "../data/MNISTtrain3(+1)8(-1)with14x14x1579.csv";
//	string testfile = "../data/MNISTt10k3(+1)8(-1)with14x14.csv";
//
//	long trainSampleDim = 0, testSampleDim = 0, trainfactorDim = 0,	testfactorDim = 0;
//	double **traindataset, **testdataset;
//	double *traindatalabel, *testdatalabel;
//
//	double **zData = MyTools::dataFromFile(trainfile, trainfactorDim, trainSampleDim, traindataset, traindatalabel);
//	double **zDate = MyTools::dataFromFile(testfile, testfactorDim, 	testSampleDim, testdataset, testdatalabel);
//	if (trainfactorDim != testfactorDim) {
//		cout << " WARNING : trainfactorDim != testfactorDim" << endl;
//		exit(-1);
//	}
//
//	MyTools::normalizeZData(traindataset, trainfactorDim, trainSampleDim);
//	MyTools::normalizeZData(testdataset, testfactorDim, testSampleDim);
//
//	//MyTools::shuffleDataSync(traindataset, trainfactorDim, trainSampleDim, traindatalabel);
//	//MyTools::shuffleDataSync(testdataset,   testfactorDim,  testSampleDim,  testdatalabel);
//
//	//---------------------------- BonteSFHwithLearningRate ---------------------------
//	string pathBonteSFHwithLearningRate = "../result/ExperimentResultInCiphertext_";
//		   pathBonteSFHwithLearningRate.append("MNISTrain_");
//		   pathBonteSFHwithLearningRate.append("BonteSFHwithXTXasG_");
//	string pathNesterovAGwithXTXasG = "../result/ExperimentResultInCiphertext_";
//		   pathNesterovAGwithXTXasG.append("MNISTrain_");
//	pathNesterovAGwithXTXasG.append("NesterovwithXTXasG_");
//
//	// Step 1. clear the former result data stored in the four*3(AUC,MLE,TIME) different files.
//	// SHOULD BE IN CONSSITENT WITH THE FILE PATH IN THE EACH ALGORITHM ! eg. "TrainAUC.csv"...
//	std::ofstream ofs;
//	ofs.open(pathBonteSFHwithLearningRate + "TrainAUC.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	ofs.open(pathBonteSFHwithLearningRate + "TrainMLE.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	ofs.open(pathBonteSFHwithLearningRate + "TestAUC.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	ofs.open(pathBonteSFHwithLearningRate + "TestMLE.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	ofs.open(pathBonteSFHwithLearningRate + "TIME.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	ofs.open(pathBonteSFHwithLearningRate + "TIMELabel.csv",std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	ofs.open(pathBonteSFHwithLearningRate + "CurrMEM.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	ofs.open(pathBonteSFHwithLearningRate + "PeakMEM.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
//
//	//ofs.open(pathNesterovAGwithXTXasG + "TrainAUC.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	//ofs.open(pathNesterovAGwithXTXasG + "TrainMLE.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	//ofs.open(pathNesterovAGwithXTXasG + "TestAUC.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	//ofs.open(pathNesterovAGwithXTXasG + "TestMLE.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	//ofs.open(pathNesterovAGwithXTXasG + "TIME.csv",			std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	//ofs.open(pathNesterovAGwithXTXasG + "TIMELabel.csv",	std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	//ofs.open(pathNesterovAGwithXTXasG + "CurrMEM.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
//	//ofs.open(pathNesterovAGwithXTXasG + "PeakMEM.csv",		std::ofstream::out | std::ofstream::trunc);	ofs.close();
//
//
//	// Step 2. Test Training and Testing ONLY one Time!
//	//         each time we choose 1579 samples from the training data set at random
//	//         use the whole test data set to validate the performance of module
//	long testSampleNum = testSampleDim;
//	long trainSampleNum = 1579;
//
//	double **traindata, **testdata;
//	double *trainlabel, *testlabel;
//
//	traindata = new double*[trainSampleNum];
//	trainlabel = new double[trainSampleNum];
//
//	testdata = testdataset;
//	testlabel = testdatalabel;
//
//
//
//		// randomly choose 1579 samples from the train data set
//		MyTools::shuffleDataSync(traindataset, trainfactorDim, trainSampleDim, traindatalabel);
//		for (long i = 0; i < trainSampleDim; ++i) {
//			traindata[i] = traindataset[i];
//			trainlabel[i] = traindatalabel[i];
//		}
//
//		//cout << "!!! Crypto: Nesterov's Accelerated Gradient Descent with 0.25XTX as Quadratic Gradient !!!" << endl;
//		//MyMethods::testCryptoNesterovWithG(traindata, trainlabel, trainfactorDim,
//		//		trainSampleDim, numIter, testdata, testlabel, testSampleDim,	pathNesterovAGwithXTXasG);
//		//cout << "### Crypto: Nesterov's Accelerated Gradient Descent with 0.25XTX as Quadratic Gradient ###" << endl;
//
//		cout << "!!! Crypto: Bonte's Simple Fixed Hessian Newton Method with Learning Rate ( 0.25XTX ) !!!"	<< endl;
//		MyMethods::testCryptoBonteSFHwithLearningRate(traindata, trainlabel, trainfactorDim,
//				trainSampleDim, numIter, testdata, testlabel, testSampleDim,	pathBonteSFHwithLearningRate);
//		cout << "### Crypto: Bonte's Simple Fixed Hessian Newton Method with Learning Rate ( 0.25XTX ) ###"	<< endl;
//
//
//
//
//
//
//	cout << endl << "END OF THE PROGRAMM" << endl;
//	return 0;
}
