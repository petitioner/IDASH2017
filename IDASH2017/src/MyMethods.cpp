/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

#include "MyMethods.h"

#include "Ciphertext.h"
#include "NTL/ZZX.h"
#include "Scheme.h"
#include "TestScheme.h"
#include "SecretKey.h"
#include "TimeUtils.h"
#include <cmath>

#include "MyTools.h"
#include <EvaluatorUtils.h>
#include <NTL/BasicThreadPool.h>
#include <NTL/RR.h>
#include <NTL/ZZ.h>


#include <iomanip>



/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

//auto hlambda = [](auto x){ return +0.49800 +0.098051*x -0.00042307*pow(x,3) -0.0000029615*pow(x,5) +0.000000022861*pow(x,7); };
auto hlambda = [](auto x){ return 1.0 / (1 + exp(-x)); };
//auto hlambda = [](auto x){ return  0.5        +0.214*x    -0.00819*pow(x,3)  +0.000165861*pow(x,5)   -0.0000011958*pow(x,7); };

// should be two of this function ?! one for IDASH, and one for MNIST. 'cause IDASH need CV, MNIST need no CV.
void MyMethods::testPlainFiveFoldCrossValidation(double** data, double* label, long factorDim, long sampleDim, long numIter, long NUMfold)
{
	// Step 1. clear the former result data stored in the four*3(AUC,MLE,TIME) different files.
	string pathNesterov                 = "./data/testPlainResultNesterov_";
	string pathNesterovWithG            = "./data/testPlainResultNesterovWithG_";
	string pathBonteSFH                 = "./data/testPlainResultBonteSFH_";
	string pathBonteSFHwithLearningRate = "./data/testPlainResultBonteSFHwithLearningRate_";

	std::ofstream ofs;
	ofs.open(pathNesterov+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterov+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterov+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathNesterovWithG+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterovWithG+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterovWithG+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathBonteSFH+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFH+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFH+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathBonteSFHwithLearningRate+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFHwithLearningRate+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFHwithLearningRate+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	// Step 2. If there is a fold cross validate, then cross the testdata


	long testSampleDim = sampleDim / NUMfold;
	long trainSampleDim = sampleDim - testSampleDim;


	double  **traindata,   **testdata;
	double  *trainlabel,   *testlabel;

	traindata = new double*[trainSampleDim];
	trainlabel= new double [trainSampleDim];

	testdata  = new double*[testSampleDim];
	testlabel = new double [testSampleDim];


	for (long fnum = 0; fnum < NUMfold; ++fnum) {
		cout << " !!! START " << fnum + 1 << " FOLD !!! " << endl;

		for (long i = 0; i < testSampleDim; ++i) {
			testdata[i] = data[fnum * testSampleDim + i];
			testlabel[i]= label[fnum * testSampleDim + i];
		}
		for (long j = 0; j < fnum; ++j) {
			for (long i = 0; i < testSampleDim; ++i) {
				traindata[j * testSampleDim + i] = data[j * testSampleDim + i];
				trainlabel[j * testSampleDim + i] = label[j * testSampleDim + i];
			}
		}
		for (long i = (fnum + 1) * testSampleDim; i < sampleDim; ++i) {
			traindata[i - testSampleDim] = data[i];
			trainlabel[i - testSampleDim] = label[i];
		}

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Nesterov's Accelerated Gradient Descent -------------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights1 = MyMethods::testPlainNesterov(traindata, trainlabel, factorDim, trainSampleDim, numIter, testdata, testlabel, testSampleDim, pathNesterov);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Nesterov with G (SFH directly by 0.25XTX) -----------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights2 = MyMethods::testPlainNesterovWithG(traindata, trainlabel, factorDim, trainSampleDim, numIter, testdata, testlabel, testSampleDim, pathNesterovWithG);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Bonte's Simplified Fixed Hessian Newton Method ------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights3 = MyMethods::testPlainBonteSFH(traindata, trainlabel, factorDim, trainSampleDim, numIter, testdata, testlabel, testSampleDim, pathBonteSFH);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Bonte's SFH with learning rate ----------------------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights4 = MyMethods::testPlainBonteSFHwithLearningRate(traindata, trainlabel, factorDim, trainSampleDim, numIter, testdata, testlabel, testSampleDim, pathBonteSFHwithLearningRate);
		cout<<"========================================== DONE! =========================================="<<endl;


		cout << " !!! STOP " << fnum + 1 << " FOLD !!! " << endl;
		cout << "------------------" << endl;

	}

	cout<<" Testing Plain Five-Fold Cross Validation is done ! "<<endl;
}

void MyMethods::testPlainTrainingAndTesting5time(double** traindata, double* trainlabel, long trainSampleDim, long factorDim, double** testdata, double* testlabel, long testSampleDim, long numIter, long NUMfold)
{
	// Step 1. clear the former result data stored in the four*3(AUC,MLE,TIME) different files.
	string pathNesterov                 = "./data/testPlainResultNesterov_";
	string pathNesterovWithG            = "./data/testPlainResultNesterovWithG_";
	string pathBonteSFH                 = "./data/testPlainResultBonteSFH_";
	string pathBonteSFHwithLearningRate = "./data/testPlainResultBonteSFHwithLearningRate_";

	std::ofstream ofs;
	ofs.open(pathNesterov+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterov+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterov+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathNesterovWithG+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterovWithG+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterovWithG+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathBonteSFH+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFH+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFH+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathBonteSFHwithLearningRate+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFHwithLearningRate+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFHwithLearningRate+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	// Step 2. If there is a fold cross validate, then cross the testdata


	for (long fnum = 0; fnum < NUMfold; ++fnum) {
		cout << " !!! START " << fnum + 1 << " FOLD !!! " << endl;

		// NO USE FOR SHUFFLE THE DATAandLABEL
		//MyTools::shuffleDataSync(traindata,factorDim, trainSampleDim, trainlabel);
		//MyTools::shuffleDataSync(testdata, factorDim,  testSampleDim,  testlabel);


		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Nesterov's Accelerated Gradient Descent -------------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights1 = MyMethods::testPlainNesterov(traindata, trainlabel, factorDim, trainSampleDim, numIter, testdata, testlabel, testSampleDim, pathNesterov);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Nesterov with G (SFH directly by 0.25XTX) -----------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights2 = MyMethods::testPlainNesterovWithG(traindata, trainlabel, factorDim, trainSampleDim, numIter, testdata, testlabel, testSampleDim, pathNesterovWithG);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Bonte's Simplified Fixed Hessian Newton Method ------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights3 = MyMethods::testPlainBonteSFH(traindata, trainlabel, factorDim, trainSampleDim, numIter, testdata, testlabel, testSampleDim, pathBonteSFH);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Bonte's SFH with learning rate ----------------------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights4 = MyMethods::testPlainBonteSFHwithLearningRate(traindata, trainlabel, factorDim, trainSampleDim, numIter, testdata, testlabel, testSampleDim, pathBonteSFHwithLearningRate);
		cout<<"========================================== DONE! =========================================="<<endl;


		cout << " !!! STOP " << fnum + 1 << " FOLD !!! " << endl;
		cout << "------------------" << endl;

	}

	cout<<" Testing Plain Training And Testing 5 times is done ! "<<endl;
}


void MyMethods::testPlainSpeedofConvergence5time(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, long NUMfold)
{
	// Step 1. clear the former result data stored in the four*3(AUC,MLE,TIME) different files.
	string pathNesterov                 = "./data/testPlainResultNesterov_";
	string pathNesterovWithG            = "./data/testPlainResultNesterovWithG_";
	string pathBonteSFH                 = "./data/testPlainResultBonteSFH_";
	string pathBonteSFHwithLearningRate = "./data/testPlainResultBonteSFHwithLearningRate_";

	std::ofstream ofs;
	ofs.open(pathNesterov+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterov+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterov+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathNesterovWithG+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterovWithG+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathNesterovWithG+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathBonteSFH+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFH+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFH+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	ofs.open(pathBonteSFHwithLearningRate+"TIME.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFHwithLearningRate+"AUC.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();
	ofs.open(pathBonteSFHwithLearningRate+"MLE.csv", std::ofstream::out | std::ofstream::trunc); ofs.close();

	// Step 2. If there is a fold cross validate, then cross the testdata


	for (long fnum = 0; fnum < NUMfold; ++fnum) {
		cout << " !!! START " << fnum + 1 << " FOLD !!! " << endl;

		// NO USE FOR SHUFFLE THE DATAandLABEL
		//MyTools::shuffleDataSync(traindata,factorDim, trainSampleDim, trainlabel);
		//MyTools::shuffleDataSync(testdata, factorDim,  testSampleDim,  testlabel);


		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Nesterov's Accelerated Gradient Descent -------------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights1 = MyMethods::testPlainNesterov(traindata, trainlabel, factorDim, trainSampleDim, numIter, traindata, trainlabel, trainSampleDim, pathNesterov);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Nesterov with G (SFH directly by 0.25XTX) -----------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights2 = MyMethods::testPlainNesterovWithG(traindata, trainlabel, factorDim, trainSampleDim, numIter, traindata, trainlabel, trainSampleDim, pathNesterovWithG);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Bonte's Simplified Fixed Hessian Newton Method ------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights3 = MyMethods::testPlainBonteSFH(traindata, trainlabel, factorDim, trainSampleDim, numIter, traindata, trainlabel, trainSampleDim, pathBonteSFH);
		cout<<"========================================== DONE! =========================================="<<endl;

		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		cout<<"------------------------- Bonte's SFH with learning rate ----------------------------------"<<endl;
		cout<<"-------------------------------------------------------------------------------------------"<<endl;
		double* weights4 = MyMethods::testPlainBonteSFHwithLearningRate(traindata, trainlabel, factorDim, trainSampleDim, numIter, traindata, trainlabel, trainSampleDim, pathBonteSFHwithLearningRate);
		cout<<"========================================== DONE! =========================================="<<endl;


		cout << " !!! STOP " << fnum + 1 << " FOLD !!! " << endl;
		cout << "------------------" << endl;

	}

	cout<<" Testing Plain Speed of Convergence 5 times is done ! "<<endl;
}


/**
 * To run the Nesterov's Accelerated Gradient Descent in plain text.
 *
 * @param  : traindata : only the train data, excluding the bias value 1
 * @param  : factorDim : the factor dimension of the traindata
 * @param  : sampleDim : the number of rows in the data
 *
 * @param  : testdata  : only the train data, excluding the bias value 1
 * @param  : testsampleDim  : the number of rows in the testdata
 *
 * @param  : resultpath: the path of the csv file to store the result
 * @return : void for now
 * @author : no one
 */
#include <unistd.h>
double* MyMethods::testPlainNesterov(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath) {
	cout<<"OPEN A FILE!"<<endl;
    // string path = "./data/testPlainNesterov.csv";
	string path = resultpath;
	ofstream openFileAUC(path+"AUC.csv",   std::ofstream::out | std::ofstream::app);	if(!openFileAUC.is_open()) cout << "Error: cannot read file" << endl;
	ofstream openFileMLE(path+"MLE.csv",   std::ofstream::out | std::ofstream::app);	if(!openFileMLE.is_open()) cout << "Error: cannot read file" << endl;
	ofstream openFileTIME(path+"TIME.csv", std::ofstream::out | std::ofstream::app);	if(!openFileTIME.is_open()) cout << "Error: cannot read file" << endl;

	openFileAUC<<"AUC";	openFileAUC.flush();
	openFileMLE<<"MLE";	openFileMLE.flush();
	openFileTIME<<"TIME";	openFileTIME.flush();

	TimeUtils timeutils;
	timeutils.start("Nesterov initializing...");

	/***** SHOULD BE DONE BEFORE! *****/
	// X = [[1]+row[1:] for row in data[:]]
	// Y = [row[0] for row in data[:]]
	// # turn y{+0,+1} to y{-1,+1}
	// Y = [2*y-1 for y in Y]
	/***** ALREADY BE DONE BEFORE! ****/

	/****************************************************************************************************************************\
	 *  Stage 2.
	 *      Step 1. Initialize Simplified Fixed Hessian Matrix
	 *      Step 2. Initialize Weight Vector (n x 1)
	 *              Setting the initial weight to 1 leads to a large input to sigmoid function,
	 *              which would cause a big problem to this algorithm when using polynomial
	 *              to substitute the sigmoid function. So, it is a good choice to set w = 0.
	 *      Step 2. Set the Maximum Iteration and Record each cost function
    \****************************************************************************************************************************/
	// w for the final weights; v for the update iteration of Nesterov’s accelerated gradient method
	double* weights = new double[factorDim]();
	double* veights = new double[factorDim]();

	double plaincor, plainauc, truecor, trueauc;
	double averplaincor = 0, averplainauc = 0, avertruecor = 0, avertrueauc = 0;


	double alpha0, alpha1, eta, gamma;
	alpha0 = 0.01;
	alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;
	/****************************************************************************************************************************\
	 *  Stage 3.
	 *      Start the Gradient Descent algorithm
	 *      Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	 *            1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	 *                   Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	 *            grad = [Y*(1 - sigm(yWTx))]T * X
    \****************************************************************************************************************************/
    //auto hlambda = [](auto x){ return 0.5+0.214*x-0.00819*pow(x,3)+0.000165861*pow(x,5)-0.0000011958*pow(x,7); };

	timeutils.stop("Nesterov initializing"); //cout<<timeutils.timeElapsed<<endl;
	cout << 0 << "-th: TIME= "<< timeutils.timeElapsed << endl;
	openFileTIME<<","<<timeutils.timeElapsed;                                                         openFileTIME.flush();

	// zData = (Y,Y@X)
	double** zData = new double*[testSampleDim];
	for(int i=0;i<testSampleDim;++i)
	{
		double* zi = new double[factorDim];
		zi[0] = testlabel[i];
		for(int j=1;j<factorDim;++j)
			zi[j] = zi[0]*testdata[i][j];
		zData[i] = zi;
	}

	cout << 0 << "-th: AUC = "<< MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) << endl;
	openFileAUC<<","<<MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileAUC.flush();

	cout << 0 << "-th: MLE = "<< MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) << endl;
	openFileMLE<<","<<MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileMLE.flush();


	for (long iter = 0; iter < numIter; ++iter) {
		timeutils.start("Nesterov : "+ to_string(iter+1)+" -th iteration");
    /****************************************************************************************************************************\
     *      Step 1. Calculate the Gradient = [Y*(1 - sigm(yWTx))]T * X
     *            # W.T * X
     *            # MXV = MX * MV
    \****************************************************************************************************************************/
		double* XW = new double[trainSampleDim]();
		for (int i = 0; i < trainSampleDim; ++i) {
			double res = 0;
			for (int j = 0; j < factorDim + 1; ++j)
				res += traindata[i][j] * weights[j];
			XW[i] = res;
		}
    /*------------# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)-----------------------------------------------------------------*/
		double* yhypothesis = new double[trainSampleDim]();
		for (int i = 0; i < trainSampleDim; ++i) {
	/*------------# the polynomial to substitute the Sigmoid function-----------------------------------------------------------*/
	/*------------#hypothesis = [ [h(wTx)] for wTx in [x[0] for x in MXW.array] ]-----------------------------------------------*/
	/*------------#h = 1 - hlambda(Y[idx]*MXV.A[idx][0])------------------------------------------------------------------------*/
	/*------------h = 1 - 1.0/(1+exp(-Y[idx]*MXV.A[idx][0]))--------------------------------------------------------------------*/
	/*------------yhypothesis.append([h*Y[idx]])--------------------------------------------------------------------------------*/
  			double sigm = 1.0 / (1 + exp(-trainlabel[i] * XW[i]));
  			sigm = hlambda(trainlabel[i] * XW[i]);
			yhypothesis[i] = trainlabel[i] * (1 - sigm);
		}

	/*------------# g = [Y@(1 - sigm(yWTx))]T * X-------------------------------------------------------------------------------*/
	/*------------Mg = MXT * Myhypothesis---------------------------------------------------------------------------------------*/
		double* grad = new double[factorDim]();
		for (int j = 0; j < factorDim; ++j) {
			double res = 0;
			for (int i = 0; i < trainSampleDim; ++i)
				res += yhypothesis[i] * traindata[i][j];

			grad[j] = res;
		}
	/****************************************************************************************************************************\
	 *      Step 2. Calculate the Hessian Matrix and its inverse
	 *            WRONG PROGRAMMING ! This will lead to update B each time by the way you dont want to.
	 *            MB_inv = MB_inv.scalarmultiply(4.0/(iter+1)+0.9)
	 *            currentMBinv = MB_inv.scalarmultiply(1.0* exp(-iter+MAX_ITER/2)/(1+exp(-iter+MAX_ITER/2))+1.0)
	 *      Step 3. Update the Weight Vector
	 *            U = inverse(B) * g
	 *            MU = currentMBinv.multiply(Mg)
	 *            MW = MW.subtract(MU)
	 *      --------------------------------------------------------------------------------------------------------------
	 *      Step 2. Update the Weight Vector using the gradients
	 *            # V is the final weight vector
	 *            # W[t+1] = V[t] - learningrate[t]*grad(V[t])
	 *            # V[t+1] = (1-eta[t])*w[t+1] + eta[t]*w[t]	 *
	\****************************************************************************************************************************/
		eta = (1 - alpha0) / alpha1;
		gamma = 1.0 / (iter + 1) / trainSampleDim;

    /*------------# should be 'plus', 'cause to compute the MLE-----------------------------------------------------------------*/
    /*------------MtmpW = MV + gamma*Mg-----------------------------------------------------------------------------------------*/
	/*------------MV = (1.0-eta)*MtmpW + (eta)*MW-------------------------------------------------------------------------------*/
	/*------------MW = MtmpW----------------------------------------------------------------------------------------------------*/
		double* tempW = new double[factorDim]();
		for (int i = 0; i < factorDim; ++i)	tempW[i] = weights[i] + gamma * grad[i];
		for (int i = 0; i < factorDim; ++i)	weights[i] = (1.0 - eta) * tempW[i] + eta * veights[i];
		for (int i = 0; i < factorDim; ++i)	veights[i] = tempW[i];

		alpha0 = alpha1;
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

		timeutils.stop("Nesterov : "+ to_string(iter+1)+" -th iteration");//cout<<timeutils.timeElapsed<<endl;
	/****************************************************************************************************************************\
	 *      Step 3. Calculate the cost function using Maximum likelihood Estimation
	 *            # weights is the final weight vector
	\****************************************************************************************************************************/
		cout << iter + 1 << "-th: TIME= "<< timeutils.timeElapsed << endl;
		cout << iter + 1 << "-th: AUC = "<< MyTools::calculateAUC(zData, weights, factorDim, testSampleDim,	truecor, trueauc) << endl;
		cout << iter + 1 << "-th: MLE = "<< MyTools::calculateMLE(zData, weights, factorDim, testSampleDim,	truecor, trueauc) << endl;

		openFileTIME<<","<<timeutils.timeElapsed;                                                         openFileTIME.flush();
		openFileAUC<<","<<MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileAUC.flush();
		openFileMLE<<","<<MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileMLE.flush();
		}

	openFileTIME<<endl; openFileTIME.flush();
	openFileAUC<<endl ; openFileAUC.flush();
	openFileMLE<<endl ; openFileMLE.flush();

	openFileTIME.close();
	openFileAUC.close();
	openFileMLE.close();
	/****************************************************************************************************************************\
	 *  Stage 4.
	 *      Test the pattern
	 *      Step 1. Calculate FN + FP
	 *      Step 2. Print some value For comparation
    \****************************************************************************************************************************/

	return weights;

}

/**
 * To run the Nesterov with G in plain text.
 * - No Cross Validation.
 * - class label Y is always the first.
 * - polynomial is always the degree 7.
 * - the learning rate is fixed. No need another two parameters.
 * - the weight vector is always initilizatied to the zero column vector.
 *
 * @param  : traindata : only the train data, excluding the bias value 1
 * @param  : factorDim : the factor dimension of the traindata
 * @param  : sampleDim : the number of rows in the data
 * @return : return the final weight column vector of the learned pattern module.
 * @author : no one
 */
#include <unistd.h>
double* MyMethods::testPlainNesterovWithG(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath) {
	cout<<"OPEN A FILE!"<<endl;
	//string path = "./data/testPlainNesterovWithG.csv";
	string path = resultpath;
	ofstream openFileAUC(path+"AUC.csv",   std::ofstream::out | std::ofstream::app);	if(!openFileAUC.is_open()) cout << "Error: cannot read file" << endl;
	ofstream openFileMLE(path+"MLE.csv",   std::ofstream::out | std::ofstream::app);	if(!openFileMLE.is_open()) cout << "Error: cannot read file" << endl;
	ofstream openFileTIME(path+"TIME.csv", std::ofstream::out | std::ofstream::app);	if(!openFileTIME.is_open()) cout << "Error: cannot read file" << endl;

	openFileAUC<<"AUC";	openFileAUC.flush();
	openFileMLE<<"MLE";	openFileMLE.flush();
	openFileTIME<<"TIME";	openFileTIME.flush();

	TimeUtils timeutils;
	timeutils.start("NesterovWithG initializing...");

	/***** SHOULD BE DONE BEFORE! *****/
	// X = [[1]+row[1:] for row in data[:]]
	// Y = [row[0] for row in data[:]]
	// # turn y{+0,+1} to y{-1,+1}
	// Y = [2*y-1 for y in Y]
	/***** ALREADY BE DONE BEFORE! ****/


	/****************************************************************************************************************************\
	 *  Stage 2.
	 *      Step 1. Initialize Simplified Fixed Hessian Matrix
	 *              # BEGIN: Bonte's Specific Order On XTX
	 *				'''
	 *				X = | X11 X12 X13 |
	 *				    | X21 X22 X23 |
	 *				    | X31 X32 X33 |
	 *				the sum of each row of (X.T * X) is a column vector as follows:
	 *				| X11 X21 X31 |   | X11+X12+X13 |
	 *				| X12 X22 X32 | * | X21+X22+X23 |
	 *				| X13 X23 X33 |   | X31+X32+X33 |
	 *				'''
	 *      Step 2. Initialize Weight Vector (n x 1)
	 *              Setting the initial weight to 1 leads to a large input to sigmoid function,
	 *              which would cause a big problem to this algorithm when using polynomial
	 *              to substitute the sigmoid function. So, it is a good choice to set w = 0.
	 *      Step 2. Set the Maximum Iteration and Record each cost function
    \****************************************************************************************************************************/
	/*--------------# return a column vector whose each element is the sum of each row of X-------------------------------------*/
	double* sumx = new double[trainSampleDim]();
	for(int i=0;i<trainSampleDim;++i)
		for(int j=0;j<factorDim;++j)
			sumx[i] += traindata[i][j];
	/*--------------# return a column vector whose each element is the sum of each row of (X.T * X)-----------------------------*/
	double* B = new double[factorDim]();
	for(int j=0;j<factorDim;++j)
		for(int i=0;i<trainSampleDim;++i)
			B[j] += traindata[i][j]*sumx[i];
	/*--------------# get the inverse of matrix MB in advance-------------------------------------------------------------------*/
	/*--------------# Be carefull with the division by zero.--------------------------------------------------------------------*/
	double epsilon = 1e-08;
	for(int i=0;i<factorDim;++i) B[i] = .25*(B[i] + epsilon);
	double* B_INV = new double[factorDim];
	for(int i=0;i<factorDim;++i) B_INV[i] = 1.0/B[i];

	/*--------------# END  : Bonte's Specific Order On .25*X.T*X----------------------------------------------------------------*/

	// w for the final weights; v for the update iteration of Nesterov’s accelerated gradient method
	double* weights = new double[factorDim]();
	double* veights = new double[factorDim]();

	double plaincor, plainauc, truecor, trueauc;
	double averplaincor = 0, averplainauc = 0, avertruecor = 0, avertrueauc = 0;


	double alpha0, alpha1, eta, gamma;
	alpha0 = 0.01;
	alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;
	/****************************************************************************************************************************\
	 *  Stage 3.
	 *      Start the Gradient Descent algorithm
	 *      Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	 *            1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	 *                   Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	 *            grad = [Y*(1 - sigm(yWTx))]T * X
    \****************************************************************************************************************************/
    //auto hlambda = [](auto x){ return 0.5+0.214*x-0.00819*pow(x,3)+0.000165861*pow(x,5)-0.0000011958*pow(x,7); };

	timeutils.stop("NesterovWithG initializing"); //cout<<timeutils.timeElapsed<<endl;
	cout << 0 << "-th: TIME= "<< timeutils.timeElapsed << endl;
	openFileTIME<<","<<timeutils.timeElapsed;                                                         openFileTIME.flush();

	// zData = (Y,Y@X)
	double** zData = new double*[testSampleDim];
	for(int i=0;i<testSampleDim;++i)
	{
		double* zi = new double[factorDim];
		zi[0] = testlabel[i];
		for(int j=1;j<factorDim;++j)
			zi[j] = zi[0]*testdata[i][j];
		zData[i] = zi;
	}

	cout << 0 << "-th: AUC = "<< MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) << endl;
	openFileAUC<<","<<MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileAUC.flush();

	cout << 0 << "-th: MLE = "<< MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) << endl;
	openFileMLE<<","<<MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileMLE.flush();

	for (long iter = 0; iter < numIter; ++iter) {
		timeutils.start("NesterovWithG : "+ to_string(iter+1)+" -th iteration");
    /****************************************************************************************************************************\
     *      Step 1. Calculate the Gradient = [Y*(1 - sigm(yWTx))]T * X
     *            # W.T * X
     *            # MXV = MX * MV
    \****************************************************************************************************************************/
		double* XW = new double[trainSampleDim]();
		for (int i = 0; i < trainSampleDim; ++i) {
			double res = 0;
			for (int j = 0; j < factorDim + 1; ++j)
				res += traindata[i][j] * weights[j];
			XW[i] = res;
		}
	/*--------------# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)-----------------------------------------------------------------*/
		double* yhypothesis = new double[trainSampleDim]();
		for (int i = 0; i < trainSampleDim; ++i) {
	/*--------------# the polynomial to substitute the Sigmoid function-----------------------------------------------------------*/
	/*--------------#hypothesis = [ [h(wTx)] for wTx in [x[0] for x in MXW.array] ]-----------------------------------------------*/
	/*--------------#h = 1 - hlambda(Y[idx]*MXV.A[idx][0])------------------------------------------------------------------------*/
	/*--------------h = 1 - 1.0/(1+exp(-Y[idx]*MXV.A[idx][0]))--------------------------------------------------------------------*/
	/*--------------yhypothesis.append([h*Y[idx]])--------------------------------------------------------------------------------*/
  			double sigm = 1.0 / (1 + exp(-trainlabel[i] * XW[i]));
  			sigm = hlambda(trainlabel[i] * XW[i]);
			yhypothesis[i] = trainlabel[i] * (1 - sigm);
		}

	/*--------------# g = [Y@(1 - sigm(yWTx))]T * X-------------------------------------------------------------------------------*/
	/*--------------Mg = MXT * Myhypothesis---------------------------------------------------------------------------------------*/
		double* grad = new double[factorDim]();
		for (int j = 0; j < factorDim; ++j) {
			double res = 0;
			for (int i = 0; i < trainSampleDim; ++i)
				res += yhypothesis[i] * traindata[i][j];

			grad[j] = res;
		}
	/****************************************************************************************************************************\
	 *      Step 2. Calculate the Hessian Matrix and its inverse
	 *            WRONG PROGRAMMING ! This will lead to update B each time by the way you dont want to.
	 *            MB_inv = MB_inv.scalarmultiply(4.0/(iter+1)+0.9)
	 *            currentMBinv = MB_inv.scalarmultiply(1.0* exp(-iter+MAX_ITER/2)/(1+exp(-iter+MAX_ITER/2))+1.0)
	 *      Step 3. Update the Weight Vector
	 *            U = inverse(B) * g
	 *            MU = currentMBinv.multiply(Mg)
	 *            MW = MW.subtract(MU)
	 *      --------------------------------------------------------------------------------------------------------------
	 *      Step 2. Update the Weight Vector using the gradients
	 *            # V is the final weight vector
	 *            # W[t+1] = V[t] - learningrate[t]*grad(V[t])
	 *            # V[t+1] = (1-eta[t])*w[t+1] + eta[t]*w[t]	 *
	\****************************************************************************************************************************/
		eta = (1 - alpha0) / alpha1;
		gamma = 1.0 / (iter + 1) / trainSampleDim;

	/*--------------# should be 'plus', 'cause to compute the MLE-----------------------------------------------------------------*/
	/*--------------MtmpW = MV + gamma*Mg-----------------------------------------------------------------------------------------*/
	/*--------------MV = (1.0-eta)*MtmpW + (eta)*MW-------------------------------------------------------------------------------*/
	/*--------------MW = MtmpW----------------------------------------------------------------------------------------------------*/
		double* tempW = new double[factorDim]();
		for (int i = 0; i < factorDim; ++i)	tempW[i] = weights[i] + (gamma+.9) * B_INV[i]*grad[i];
		for (int i = 0; i < factorDim; ++i)	weights[i] = (1.0 - eta) * tempW[i] + eta * veights[i];
		for (int i = 0; i < factorDim; ++i)	veights[i] = tempW[i];

		alpha0 = alpha1;
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

		timeutils.stop("NesterovWithG : "+ to_string(iter+1)+" -th iteration");//cout<<timeutils.timeElapsed<<endl;
	/****************************************************************************************************************************\
	 *      Step 3. Calculate the cost function using Maximum likelihood Estimation
	 *            # weights is the final weight vector
	\****************************************************************************************************************************/
		// zData = (Y,Y@X)

		cout << iter + 1 << "-th: TIME= "<< timeutils.timeElapsed << endl;
		cout << iter + 1 << "-th: AUC = "<< MyTools::calculateAUC(zData, weights, factorDim, testSampleDim,	truecor, trueauc) << endl;
		cout << iter + 1 << "-th: MLE = "<< MyTools::calculateMLE(zData, weights, factorDim, testSampleDim,	truecor, trueauc) << endl;

		openFileTIME<<","<<timeutils.timeElapsed;                                                         openFileTIME.flush();
		openFileAUC<<","<<MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileAUC.flush();
		openFileMLE<<","<<MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileMLE.flush();
		}

	openFileTIME<<endl; openFileTIME.flush();
	openFileAUC<<endl ; openFileAUC.flush();
	openFileMLE<<endl ; openFileMLE.flush();

	openFileTIME.close();
	openFileAUC.close();
	openFileMLE.close();
	/****************************************************************************************************************************\
	 *  Stage 4.
	 *      Test the pattern
	 *      Step 1. Calculate FN + FP
	 *      Step 2. Print some value For comparation
    \****************************************************************************************************************************/

	return weights;

}

/**
 * To run the Bonte's Simplified Fixed Hessian Newton Method.
 * - No Cross Validation.
 * - class label Y is always the first.
 * - polynomial is always the degree 7.
 * - the learning rate is fixed. No need another two parameters.
 * - the weight vector is always initilizatied to the zero column vector.
 *
 * @param  : traindata : only the train data, excluding the bias value 1
 * @param  : factorDim : the factor dimension of the traindata
 * @param  : sampleDim : the number of rows in the data
 * @return : return the final weight column vector of the learned pattern module.
 * @author : no one
 */
#include <unistd.h>
double* MyMethods::testPlainBonteSFH(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath) {
	cout<<"OPEN A FILE!"<<endl;
	//string path = "./data/testPlainBonteSFH.csv";
	string path = resultpath;
	ofstream openFileAUC(path+"AUC.csv",   std::ofstream::out | std::ofstream::app);	if(!openFileAUC.is_open()) cout << "Error: cannot read file" << endl;
	ofstream openFileMLE(path+"MLE.csv",   std::ofstream::out | std::ofstream::app);	if(!openFileMLE.is_open()) cout << "Error: cannot read file" << endl;
	ofstream openFileTIME(path+"TIME.csv", std::ofstream::out | std::ofstream::app);	if(!openFileTIME.is_open()) cout << "Error: cannot read file" << endl;

	openFileAUC<<"AUC";	openFileAUC.flush();
	openFileMLE<<"MLE";	openFileMLE.flush();
	openFileTIME<<"TIME";	openFileTIME.flush();

	TimeUtils timeutils;
	timeutils.start("BonteSFH initializing...");

	/***** SHOULD BE DONE BEFORE! *****/
	// X = [[1]+row[1:] for row in data[:]]
	// Y = [row[0] for row in data[:]]
	// # turn y{+0,+1} to y{-1,+1}
	// Y = [2*y-1 for y in Y]
	/***** ALREADY BE DONE BEFORE! ****/



	/****************************************************************************************************************************\
	 *  Stage 2.
	 *      Step 1. Initialize Simplified Fixed Hessian Matrix
	 *              # BEGIN: Bonte's Specific Order On XTX
	 *				'''
	 *				X = | X11 X12 X13 |
	 *				    | X21 X22 X23 |
	 *				    | X31 X32 X33 |
	 *				the sum of each row of (X.T * X) is a column vector as follows:
	 *				| X11 X21 X31 |   | X11+X12+X13 |
	 *				| X12 X22 X32 | * | X21+X22+X23 |
	 *				| X13 X23 X33 |   | X31+X32+X33 |
	 *				'''
	 *      Step 2. Initialize Weight Vector (n x 1)
	 *              Setting the initial weight to 1 leads to a large input to sigmoid function,
	 *              which would cause a big problem to this algorithm when using polynomial
	 *              to substitute the sigmoid function. So, it is a good choice to set w = 0.
	 *      Step 2. Set the Maximum Iteration and Record each cost function
    \****************************************************************************************************************************/
	/*--------------# return a column vector whose each element is the sum of each row of X-------------------------------------*/
	double* sumx = new double[trainSampleDim]();
	for(int i=0;i<trainSampleDim;++i)
		for(int j=0;j<factorDim;++j)
			sumx[i] += traindata[i][j];
	/*--------------# return a column vector whose each element is the sum of each row of (X.T * X)-----------------------------*/
	double* B = new double[factorDim]();
	for(int j=0;j<factorDim;++j)
		for(int i=0;i<trainSampleDim;++i)
			B[j] += traindata[i][j]*sumx[i];
	/*--------------# get the inverse of matrix MB in advance-------------------------------------------------------------------*/
	/*--------------# Be carefull with the division by zero.--------------------------------------------------------------------*/
	double epsilon = 1e-08;
	for(int i=0;i<factorDim;++i) B[i] = .25*(B[i] + epsilon);
	double* B_INV = new double[factorDim];
	for(int i=0;i<factorDim;++i) B_INV[i] = 1.0/B[i];
	/*--------------# END  : Bonte's Specific Order On .25*X.T*X----------------------------------------------------------------*/

	// w for the final weights; v for the update iteration of Nesterov’s accelerated gradient method
	double* weights = new double[factorDim]();

	double plaincor, plainauc, truecor, trueauc;
	double averplaincor = 0, averplainauc = 0, avertruecor = 0, avertrueauc = 0;


	/****************************************************************************************************************************\
	 *  Stage 3.
	 *      Start the Gradient Descent algorithm
	 *      Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	 *            1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	 *                   Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	 *            grad = [Y*(1 - sigm(yWTx))]T * X
    \****************************************************************************************************************************/
    //auto hlambda = [](auto x){ return 0.5+0.214*x-0.00819*pow(x,3)+0.000165861*pow(x,5)-0.0000011958*pow(x,7); };

	timeutils.stop("BonteSFH initializing"); //cout<<timeutils.timeElapsed<<endl;
	cout << 0 << "-th: TIME= "<< timeutils.timeElapsed << endl;
	openFileTIME<<","<<timeutils.timeElapsed;                                                         openFileTIME.flush();

	// zData = (Y,Y@X)
	double** zData = new double*[testSampleDim];
	for(int i=0;i<testSampleDim;++i)
	{
		double* zi = new double[factorDim];
		zi[0] = testlabel[i];
		for(int j=1;j<factorDim;++j)
			zi[j] = zi[0]*testdata[i][j];
		zData[i] = zi;
	}

	cout << 0 << "-th: AUC = "<< MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) << endl;
	openFileAUC<<","<<MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileAUC.flush();

	cout << 0 << "-th: MLE = "<< MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) << endl;
	openFileMLE<<","<<MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileMLE.flush();


	for (long iter = 0; iter < numIter; ++iter) {
		timeutils.start("BonteSFH : "+ to_string(iter+1)+" -th iteration");
    /****************************************************************************************************************************\
     *      Step 1. Calculate the Gradient = [Y*(1 - sigm(yWTx))]T * X
     *            # W.T * X
     *            # MXV = MX * MV
    \****************************************************************************************************************************/
		double* XW = new double[trainSampleDim]();
		for (int i = 0; i < trainSampleDim; ++i) {
			double res = 0;
			for (int j = 0; j < factorDim + 1; ++j)
				res += traindata[i][j] * weights[j];
			XW[i] = res;
		}
	/*--------------# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)-----------------------------------------------------------------*/
		double* yhypothesis = new double[trainSampleDim]();
		for (int i = 0; i < trainSampleDim; ++i) {
	/*--------------# the polynomial to substitute the Sigmoid function-----------------------------------------------------------*/
	/*--------------#hypothesis = [ [h(wTx)] for wTx in [x[0] for x in MXW.array] ]-----------------------------------------------*/
	/*--------------#h = 1 - hlambda(Y[idx]*MXV.A[idx][0])------------------------------------------------------------------------*/
	/*--------------h = 1 - 1.0/(1+exp(-Y[idx]*MXV.A[idx][0]))--------------------------------------------------------------------*/
	/*--------------yhypothesis.append([h*Y[idx]])--------------------------------------------------------------------------------*/
  			double sigm = 1.0 / (1 + exp(-trainlabel[i] * XW[i]));
  			sigm = hlambda(trainlabel[i] * XW[i]);
			yhypothesis[i] = trainlabel[i] * (1 - sigm);
		}

	/*--------------# g = [Y@(1 - sigm(yWTx))]T * X-------------------------------------------------------------------------------*/
	/*--------------Mg = MXT * Myhypothesis---------------------------------------------------------------------------------------*/
		double* grad = new double[factorDim]();
		for (int j = 0; j < factorDim; ++j) {
			double res = 0;
			for (int i = 0; i < trainSampleDim; ++i)
				res += yhypothesis[i] * traindata[i][j];

			grad[j] = res;
		}
	/****************************************************************************************************************************\
	 *      Step 2. Calculate the Hessian Matrix and its inverse
	 *            WRONG PROGRAMMING ! This will lead to update B each time by the way you dont want to.
	 *            MB_inv = MB_inv.scalarmultiply(4.0/(iter+1)+0.9)
	 *            currentMBinv = MB_inv.scalarmultiply(1.0* exp(-iter+MAX_ITER/2)/(1+exp(-iter+MAX_ITER/2))+1.0)
	 *      Step 3. Update the Weight Vector
	 *            U = inverse(B) * g
	 *            MU = currentMBinv.multiply(Mg)
	 *            MW = MW.subtract(MU)
	 *      --------------------------------------------------------------------------------------------------------------
	 *      Step 2. Update the Weight Vector using the gradients
	 *            # V is the final weight vector
	 *            # W[t+1] = V[t] - learningrate[t]*grad(V[t])
	 *            # V[t+1] = (1-eta[t])*w[t+1] + eta[t]*w[t]	 *
	\****************************************************************************************************************************/

	/*--------------# should be 'plus', 'cause to compute the MLE-----------------------------------------------------------------*/
	/*--------------MtmpW = MV + gamma*Mg-----------------------------------------------------------------------------------------*/
	/*--------------MV = (1.0-eta)*MtmpW + (eta)*MW-------------------------------------------------------------------------------*/
	/*--------------MW = MtmpW----------------------------------------------------------------------------------------------------*/

		for (int i = 0; i < factorDim; ++i) weights[i] = weights[i] + B_INV[i]*grad[i];


		timeutils.stop("BonteSFH : "+ to_string(iter+1)+" -th iteration");//cout<<timeutils.timeElapsed<<endl;
	/****************************************************************************************************************************\
	 *      Step 3. Calculate the cost function using Maximum likelihood Estimation
	 *            # weights is the final weight vector
	\****************************************************************************************************************************/
		// zData = (Y,Y@X)
		cout << iter + 1 << "-th: TIME= "<< timeutils.timeElapsed << endl;
		cout << iter + 1 << "-th: AUC = "<< MyTools::calculateAUC(zData, weights, factorDim, testSampleDim,	truecor, trueauc) << endl;
		cout << iter + 1 << "-th: MLE = "<< MyTools::calculateMLE(zData, weights, factorDim, testSampleDim,	truecor, trueauc) << endl;

		openFileTIME<<","<<timeutils.timeElapsed;                                                         openFileTIME.flush();
		openFileAUC<<","<<MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileAUC.flush();
		openFileMLE<<","<<MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileMLE.flush();
		}

	openFileTIME<<endl; openFileTIME.flush();
	openFileAUC<<endl ; openFileAUC.flush();
	openFileMLE<<endl ; openFileMLE.flush();

	openFileTIME.close();
	openFileAUC.close();
	openFileMLE.close();
	/****************************************************************************************************************************\
	 *  Stage 4.
	 *      Test the pattern
	 *      Step 1. Calculate FN + FP
	 *      Step 2. Print some value For comparation
    \****************************************************************************************************************************/

	return weights;

}

/**
 * To run the Bonte's Simplified Fixed Hessian Newton Method with Learning Rate.
 * - No Cross Validation.
 * - class label Y is always the first.
 * - polynomial is always the degree 7.
 * - the learning rate is fixed. No need another two parameters.
 * - the weight vector is always initilizatied to the zero column vector.
 *
 * @param  : traindata : only the train data, excluding the bias value 1
 * @param  : factorDim : the factor dimension of the traindata
 * @param  : sampleDim : the number of rows in the data
 * @return : return the final weight column vector of the learned pattern module.
 * @author : no one
 */
#include <unistd.h>
double* MyMethods::testPlainBonteSFHwithLearningRate(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath) {
	cout<<"OPEN A FILE!"<<endl;
	//string path = "./data/testPlainBonteSFHwithLearningRate.csv";
	string path = resultpath;
	ofstream openFileAUC(path+"AUC.csv",   std::ofstream::out | std::ofstream::app);	if(!openFileAUC.is_open()) cout << "Error: cannot read file" << endl;
	ofstream openFileMLE(path+"MLE.csv",   std::ofstream::out | std::ofstream::app);	if(!openFileMLE.is_open()) cout << "Error: cannot read file" << endl;
	ofstream openFileTIME(path+"TIME.csv", std::ofstream::out | std::ofstream::app);	if(!openFileTIME.is_open()) cout << "Error: cannot read file" << endl;

	openFileAUC<<"AUC";	openFileAUC.flush();
	openFileMLE<<"MLE";	openFileMLE.flush();
	openFileTIME<<"TIME";	openFileTIME.flush();
	TimeUtils timeutils;
	timeutils.start("BonteSFHwithLearningRate initializing...");

	/***** SHOULD BE DONE BEFORE! *****/
	// X = [[1]+row[1:] for row in data[:]]
	// Y = [row[0] for row in data[:]]
	// # turn y{+0,+1} to y{-1,+1}
	// Y = [2*y-1 for y in Y]
	/***** ALREADY BE DONE BEFORE! ****/



	/****************************************************************************************************************************\
	 *  Stage 2.
	 *      Step 1. Initialize Simplified Fixed Hessian Matrix
	 *              # BEGIN: Bonte's Specific Order On XTX
	 *				'''
	 *				X = | X11 X12 X13 |
	 *				    | X21 X22 X23 |
	 *				    | X31 X32 X33 |
	 *				the sum of each row of (X.T * X) is a column vector as follows:
	 *				| X11 X21 X31 |   | X11+X12+X13 |
	 *				| X12 X22 X32 | * | X21+X22+X23 |
	 *				| X13 X23 X33 |   | X31+X32+X33 |
	 *				'''
	 *      Step 2. Initialize Weight Vector (n x 1)
	 *              Setting the initial weight to 1 leads to a large input to sigmoid function,
	 *              which would cause a big problem to this algorithm when using polynomial
	 *              to substitute the sigmoid function. So, it is a good choice to set w = 0.
	 *      Step 2. Set the Maximum Iteration and Record each cost function
    \****************************************************************************************************************************/
	/*--------------# return a column vector whose each element is the sum of each row of X-------------------------------------*/
	double* sumx = new double[trainSampleDim]();
	for(int i=0;i<trainSampleDim;++i)
		for(int j=0;j<factorDim;++j)
			sumx[i] += traindata[i][j];
	/*--------------# return a column vector whose each element is the sum of each row of (X.T * X)-----------------------------*/
	double* B = new double[factorDim]();
	for(int j=0;j<factorDim;++j)
		for(int i=0;i<trainSampleDim;++i)
			B[j] += traindata[i][j]*sumx[i];
	/*--------------# get the inverse of matrix MB in advance-------------------------------------------------------------------*/
	/*--------------# Be carefull with the division by zero.--------------------------------------------------------------------*/
	double epsilon = 1e-08;
	for(int i=0;i<factorDim;++i) B[i] = .25*(B[i] + epsilon);
	double* B_INV = new double[factorDim];
	for(int i=0;i<factorDim;++i) B_INV[i] = 1.0/B[i];
	/*--------------# END  : Bonte's Specific Order On .25*X.T*X----------------------------------------------------------------*/

	// w for the final weights; v for the update iteration of Nesterov’s accelerated gradient method
	double* weights = new double[factorDim]();

	double plaincor, plainauc, truecor, trueauc;
	double averplaincor = 0, averplainauc = 0, avertruecor = 0, avertrueauc = 0;

	/****************************************************************************************************************************\
	 *  Stage 3.
	 *      Start the Gradient Descent algorithm
	 *      Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))
	 *            1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))
	 *                   Prob(y= I|x) = 1/(1+exp(-I*W.T*x))
	 *            grad = [Y*(1 - sigm(yWTx))]T * X
    \****************************************************************************************************************************/
    //auto hlambda = [](auto x){ return 0.5+0.214*x-0.00819*pow(x,3)+0.000165861*pow(x,5)-0.0000011958*pow(x,7); };

	timeutils.stop("BonteSFHwithLearningRate initializing"); //cout<<timeutils.timeElapsed<<endl;
	cout << 0 << "-th: TIME= "<< timeutils.timeElapsed << endl;
	openFileTIME<<","<<timeutils.timeElapsed;                                                         openFileTIME.flush();

	// zData = (Y,Y@X)
	double** zData = new double*[testSampleDim];
	for(int i=0;i<testSampleDim;++i)
	{
		double* zi = new double[factorDim];
		zi[0] = testlabel[i];
		for(int j=1;j<factorDim;++j)
			zi[j] = zi[0]*testdata[i][j];
		zData[i] = zi;
	}

	cout << 0 << "-th: AUC = "<< MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) << endl;
	openFileAUC<<","<<MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileAUC.flush();

	cout << 0 << "-th: MLE = "<< MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) << endl;
	openFileMLE<<","<<MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileMLE.flush();

	for (long iter = 0; iter < numIter; ++iter) {
		timeutils.start("BonteSFHwithLearningRate : "+ to_string(iter+1)+" -th iteration");
    /****************************************************************************************************************************\
     *      Step 1. Calculate the Gradient = [Y*(1 - sigm(yWTx))]T * X
     *            # W.T * X
     *            # MXV = MX * MV
    \****************************************************************************************************************************/
		double* XW = new double[trainSampleDim]();
		for (int i = 0; i < trainSampleDim; ++i) {
			double res = 0;
			for (int j = 0; j < factorDim + 1; ++j)
				res += traindata[i][j] * weights[j];
			XW[i] = res;
		}
	/*--------------# [Y@(1 - sigm(Y@WT*X))]     (Y=1 if h(x)>.5)-----------------------------------------------------------------*/
		double* yhypothesis = new double[trainSampleDim]();
		for (int i = 0; i < trainSampleDim; ++i) {
	/*--------------# the polynomial to substitute the Sigmoid function-----------------------------------------------------------*/
	/*--------------#hypothesis = [ [h(wTx)] for wTx in [x[0] for x in MXW.array] ]-----------------------------------------------*/
	/*--------------#h = 1 - hlambda(Y[idx]*MXV.A[idx][0])------------------------------------------------------------------------*/
	/*--------------h = 1 - 1.0/(1+exp(-Y[idx]*MXV.A[idx][0]))--------------------------------------------------------------------*/
	/*--------------yhypothesis.append([h*Y[idx]])--------------------------------------------------------------------------------*/
  			double sigm = 1.0 / (1 + exp(-trainlabel[i] * XW[i]));
  			sigm = hlambda(trainlabel[i] * XW[i]);
			yhypothesis[i] = trainlabel[i] * (1 - sigm);
		}

	/*--------------# g = [Y@(1 - sigm(yWTx))]T * X-------------------------------------------------------------------------------*/
	/*--------------Mg = MXT * Myhypothesis---------------------------------------------------------------------------------------*/
		double* grad = new double[factorDim]();
		for (int j = 0; j < factorDim; ++j) {
			double res = 0;
			for (int i = 0; i < trainSampleDim; ++i)
				res += yhypothesis[i] * traindata[i][j];

			grad[j] = res;
		}
	/****************************************************************************************************************************\
	 *      Step 2. Calculate the Hessian Matrix and its inverse
	 *            WRONG PROGRAMMING ! This will lead to update B each time by the way you dont want to.
	 *            MB_inv = MB_inv.scalarmultiply(4.0/(iter+1)+0.9)
	 *            currentMBinv = MB_inv.scalarmultiply(1.0* exp(-iter+MAX_ITER/2)/(1+exp(-iter+MAX_ITER/2))+1.0)
	 *      Step 3. Update the Weight Vector
	 *            U = inverse(B) * g
	 *            MU = currentMBinv.multiply(Mg)
	 *            MW = MW.subtract(MU)
	 *      --------------------------------------------------------------------------------------------------------------
	 *      Step 2. Update the Weight Vector using the gradients
	 *            # V is the final weight vector
	 *            # W[t+1] = V[t] - learningrate[t]*grad(V[t])
	 *            # V[t+1] = (1-eta[t])*w[t+1] + eta[t]*w[t]	 *
	\****************************************************************************************************************************/


	/*--------------# should be 'plus', 'cause to compute the MLE-----------------------------------------------------------------*/
	/*--------------MtmpW = MV + gamma*Mg-----------------------------------------------------------------------------------------*/
	/*--------------MV = (1.0-eta)*MtmpW + (eta)*MW-------------------------------------------------------------------------------*/
	/*--------------MW = MtmpW----------------------------------------------------------------------------------------------------*/
	/*--------------# rescale Iter into [-10,+10] using x = (10--10)*(iter-1)/(MaxIter-1) + -10-----------------------------------*/
	/*--------------# the learning rate = 2 - y = 2 - 1./(1+e^(-x)) = 1 + 1/(e^x + 1)---------------------------------------------*/
    /*--------------sigmoidx = 20*(iter+1 -1)/(MAX_ITER -1) -10-------------------------------------------------------------------*/
    /*--------------learningrate = 1.0 + 1.0/(exp(sigmoidx) + 1.0)----------------------------------------------------------------*/
		double sigmoidx = 20*(iter+1 -1)/(numIter -1) -10;
		double learningrate = 1.0 + 1.0/(exp(sigmoidx) + 1.0);
		for (int i = 0; i < factorDim; ++i)	weights[i] = weights[i] + learningrate * B_INV[i]*grad[i];

		timeutils.stop("BonteSFHwithLearningRate : "+ to_string(iter+1)+" -th iteration");//cout<<timeutils.timeElapsed<<endl;
	/****************************************************************************************************************************\
	 *      Step 3. Calculate the cost function using Maximum likelihood Estimation
	 *            # weights is the final weight vector
	\****************************************************************************************************************************/
		// zData = (Y,Y@X)

		cout << iter + 1 << "-th: TIME= "<< timeutils.timeElapsed << endl;
		cout << iter + 1 << "-th: AUC = "<< MyTools::calculateAUC(zData, weights, factorDim, testSampleDim,	truecor, trueauc) << endl;
		cout << iter + 1 << "-th: MLE = "<< MyTools::calculateMLE(zData, weights, factorDim, testSampleDim,	truecor, trueauc) << endl;

		openFileTIME<<","<<timeutils.timeElapsed;                                                         openFileTIME.flush();
		openFileAUC<<","<<MyTools::calculateAUC(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileAUC.flush();
		openFileMLE<<","<<MyTools::calculateMLE(zData, weights, factorDim, testSampleDim, truecor, trueauc) ; openFileMLE.flush();
		}

	openFileTIME<<endl; openFileTIME.flush();
	openFileAUC<<endl ; openFileAUC.flush();
	openFileMLE<<endl ; openFileMLE.flush();

	openFileTIME.close();
	openFileAUC.close();
	openFileMLE.close();
	/****************************************************************************************************************************\
	 *  Stage 4.
	 *      Test the pattern
	 *      Step 1. Calculate FN + FP
	 *      Step 2. Print some value For comparation
    \****************************************************************************************************************************/

	return weights;

}


/*==========================================================================================================================================================================================================*\
 *                                                                                   To Run The Algorithms In Crypto-Text                                                                                   *
\*==========================================================================================================================================================================================================*/

#include <iomanip>

double* MyMethods::testCryptoBonteSFHwithLearningRate(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath)
{
	long fdimBits = (long)ceil(log2(factorDim));              //ceil(x) : Rounds x upward, returning the smallest integral value that is not less than x.
	long sdimBits = (long)ceil(log2(trainSampleDim));         //log2(x) : Returns the binary (base-2) logarithm of x.
/*
	cnum - is number of ciphertexts we need to pack the full dataset.
	wBits - is number of precision bits we have when encrypting the dataset.
	  pBits - is number of precision bits we have for constants encoding (as constants encoding almost have no errors, we can make pBits smaller than wBits)
	  ?lBits - is the number of bits above wBits for the final result (if result is bigger than 1, then we cannot fit it in wBits, we need more)
	aBits - is some parameter for sigmoid evaluation and normally is 2 or 3.
	kdeg - is degree of sigmoid approximation
	slots - number of slots in one ciphertext and is not bigger than N/2 , probably for translating complex to real.
	sBits = log2(slots) from the code
	batch - how many features we pack in one ciphertext.

 */
	long wBits = 30;                                          // Δ (delta)
	long pBits = 20;
	long lBits = 5;
	long aBits = 3;
	long kdeg = 7;
	long kBits = (long)ceil(log2(kdeg));                      // N is the dimension of the Ring; 2^sdimBits is the size of trainSampleDim

	//long MAX_ITER = numIter;	
	//          final result    + iteration * each...
	
	//logQ = 1200;
	cout << "logQ = " << logQ << endl;
	// Do Not use this logQ to bootstrap
	long logN = MyTools::suggestLogN(80, logQ);  // it should be the Security Parameter λ

	//////////////////////// NEED MORE CAREFULLY DEAL WITH THIS ///////////////////////////
	/////////////// COULD DIVIDE THE WHOLE DATA SET INTO SEVERAL BLOCKS ///////////////////
	// BR MORE CAREHULL OF THE WIDE batch(2^bBits) AND LENGTH trainSampleDim(2^sdimBits) //
	///////////////////////////////////////////////////////////////////////////////////////
	long bBits = min(logN - 1 - sdimBits, fdimBits);          // 2^batchBits = min( 2^logN / 2^sdimBits / 2, 2^fdimBits ) = min( N/2 /n, factorDim ) ;
	//bBits = 13 - sdimBits; //// SO THAT CAN BOOTSTRAP!!
	// make logN = 13, then set down the logQ according to the Security Parameter λ             XXXX WRONG STATEMENT
	bBits = 1; // should be changeable !!!
	if (bBits + sdimBits <= 13) bBits = 13-sdimBits;
	else {
		cout << "WARNING IN THE CHOICE OF bBits OR THE DATASET IS TOO LARGE!" << endl;
		exit(-1);
	}

	// consider the simple way: bBits = min(logN-sdimBits, fdimBits), defined as the number of how many columns in a cipher-text.
	// To be N/2 is because of the translation from complex number to real number.
	long batch = 1 << bBits;         // Basically, batch is the Number of several factor dimensions.
	long sBits = sdimBits + bBits;   // 2^sBits = 2^sdimBits * 2^bBits = ceil2(trainSampleDim) * batch
	long slots =  1 << sBits;        //
	long cnum = (long)ceil((double)factorDim / batch);  // To Divide the whole Train Data into Several Batches (cnum Ciphertexts).

	cout << "batch = " << batch << ", slots = " << slots << ", cnum = " << cnum << endl;
	cout<<"logQ = "<<logQ<<", logN = "<<logN<<", sdimBits = "<<sdimBits<<", fdimBits = "<<fdimBits<<endl;


	//string path = "./data/testCryptoNesterovWithG_";
	string path = resultpath;
	ofstream openFileTrainAUC(path+"TrainAUC.csv",   std::ofstream::out | std::ofstream::app);
	ofstream openFileTrainMLE(path+"TrainMLE.csv",   std::ofstream::out | std::ofstream::app);
	ofstream openFileTestAUC(path+"TestAUC.csv",   std::ofstream::out | std::ofstream::app);
	ofstream openFileTestMLE(path+"TestMLE.csv",   std::ofstream::out | std::ofstream::app);
	ofstream openFileTIME(path+"TIME.csv", std::ofstream::out | std::ofstream::app);
	ofstream openFileTIMELabel(path+"TIMELabel.csv", std::ofstream::out | std::ofstream::app);
	ofstream openFileCurrMEM(path+"CurrMEM.csv", std::ofstream::out | std::ofstream::app);
	ofstream openFilePeakMEM(path+"PeakMEM.csv", std::ofstream::out | std::ofstream::app);

	if(!openFileTrainAUC.is_open()) cout << "Error: cannot read Train AUC file" << endl;
	if(!openFileTrainMLE.is_open()) cout << "Error: cannot read Train MLE file" << endl;
	if(!openFileTestAUC.is_open())  cout << "Error: cannot read Test AUC file" << endl;
	if(!openFileTestMLE.is_open())  cout << "Error: cannot read Test MLE file" << endl;
	if(!openFileTIME.is_open())     cout << "Error: cannot read TIME file" << endl;
	if(!openFileTIMELabel.is_open())cout << "Error: cannot read TIME Label file" << endl;
	if(!openFileTIMELabel.is_open())cout << "Error: cannot read TIME Label file" << endl;
	if(!openFileCurrMEM.is_open())  cout << "Error: cannot read Current MEMORY file" << endl;
	if(!openFilePeakMEM.is_open())  cout << "Error: cannot read Peak MEMORY file" << endl;

	openFileTrainAUC<<"TrainAUC";	openFileTrainAUC.flush();
	openFileTrainMLE<<"TrainMLE";	openFileTrainMLE.flush();
	openFileTestAUC<<"TestAUC";	openFileTestAUC.flush();
	openFileTestMLE<<"TestMLE";	openFileTestMLE.flush();
	openFileTIME<<"TIME";	    openFileTIME.flush();
	openFileTIMELabel<<"TIMELabel";	openFileTIMELabel.flush();
	openFileCurrMEM<<"MEMORY(GB)";	openFileCurrMEM.flush();
	openFilePeakMEM<<"MEMORY(GB)";	openFilePeakMEM.flush();


	TimeUtils timeutils;
	timeutils.start("Scheme generating...");
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);
	scheme.addLeftRotKeys(secretKey);
	scheme.addRightRotKeys(secretKey);
	timeutils.stop("Scheme generation");
	cout << 0 << "-th: TIME= "<< timeutils.timeElapsed << endl;
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Scheme generating";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFilePeakMEM.flush();

	cout << "NOW THE PEAK RSS IS: " << ( MyTools::getPeakRSS() >> 20 ) << endl;
    timeutils.start("Bootstrap Key generating");
    long bootlogq = 40  +10;
    //scheme.addBootKey(secretKey, logn, logq+logI);
    scheme.addBootKey(secretKey, sBits, bootlogq + 4);
    timeutils.stop("Bootstrap Key generated");
    openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
    openFileTIMELabel<<","<<"Bootstrap Key generating";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();
	cout << "NOW THE CURR RSS IS: " << ( MyTools::getCurrentRSS() >> 20 ) << endl;
	cout << "NOW THE PEAK RSS IS: " << ( MyTools::getPeakRSS() >> 20 ) << endl;

	//CipherGD cipherGD(scheme, secretKey);

	// Basically, rpoly is used to calculate the sum{row}
	timeutils.start("Polynomial generating...");
	long np = ceil((pBits + logQ + logN + 2)/59.);
	uint64_t* rpoly = new uint64_t[np << logN];
	/* cipherGD.generateAuxPoly(rpoly, slots, batch, pBits); */
	complex<double>* pvals = new complex<double> [slots];
	for (long j = 0; j < slots; j += batch) {
		pvals[j] = 1.0;
	}
	ZZ* msg = new ZZ[N];
	scheme.ring.encode(msg, pvals, slots, pBits);
	scheme.ring.CRT(rpoly, msg, np);
	delete[] pvals;
	delete[] msg;
	/* cipherGD.generateAuxPoly(rpoly, slots, batch, pBits); */
	timeutils.stop("Polynomial generation");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Polynomial generating";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

	double* cwData = new double[factorDim]; //Current Iteration Weights Data
	double* cvData = new double[factorDim];


	Ciphertext* encTrainData = new Ciphertext[cnum];
	Ciphertext* encTrainLabel= new Ciphertext[cnum];

	Ciphertext* encVData = new Ciphertext[cnum];


	/* - - - - - - - - - - - - - - - - - - - - - - - - Client and Server - - - - - - - - - - - - - - - - - - - - - - - - */


	// zData = (Y,Y@X)
	double** zDataTrain = new double*[trainSampleDim];
	for(int i=0;i<trainSampleDim;++i)
	{
		double* zi = new double[factorDim];
		zi[0] = trainlabel[i];
		for(int j=1;j<factorDim;++j)
			zi[j] = zi[0]*traindata[i][j];
		zDataTrain[i] = zi;
	}
	// zDataTest is only used for Cross-Validation test, not necessary for training LG model.
	// zData = (Y,Y@X)
	double** zDataTest = new double*[testSampleDim];
	for(int i=0;i<testSampleDim;++i)
	{
		double* zi = new double[factorDim];
		zi[0] = testlabel[i];
		for(int j=1;j<factorDim;++j)
			zi[j] = zi[0]*testdata[i][j];
		zDataTest[i] = zi;
	}

	/* cipherGD.encZData(encZData, zDataTrain, slots, factorDim, trainSampleDim, batch, cnum, wBits, logQ);  */
	timeutils.start("Encrypting trainlabel...");
	// encrypt the trainlabel
	complex<double>* pzLabel = new complex<double>[slots];
	for (long i = 0; i < cnum - 1; ++i) {
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l) {
				pzLabel[batch * j + l].real(trainlabel[j]);
				pzLabel[batch * j + l].imag(0);
			}
		}
		scheme.encrypt(encTrainLabel[i], pzLabel, slots, wBits, logQ);
	}
	long rest = factorDim - batch * (cnum - 1);
	for (long j = 0; j < trainSampleDim; ++j) {
		for (long l = 0; l < rest; ++l) {
			pzLabel[batch * j + l].real(trainlabel[j]);
			pzLabel[batch * j + l].imag(0);
		}
		for (long l = rest; l < batch; ++l) {
			//pzDataLabel[batch * j + l] = 0;
			pzLabel[batch * j + l].real(0);
			pzLabel[batch * j + l].imag(0);
		}
	}
	scheme.encrypt(encTrainLabel[cnum - 1], pzLabel, slots, wBits, logQ);
	delete[] pzLabel;
	timeutils.stop("trainlabel encryption");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Encrypting trainlabel";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

	timeutils.start("Encrypting traindata...");
	// encrypt the traindata
	complex<double>* pzData = new complex<double>[slots];
	for (long i = 0; i < cnum - 1; ++i) {
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l) {
				pzData[batch * j + l].real(traindata[j][batch * i + l]);
				pzData[batch * j + l].imag(0);
			}
		}
		scheme.encrypt(encTrainData[i], pzData, slots, wBits, logQ);
	}
	rest = factorDim - batch * (cnum - 1);
	for (long j = 0; j < trainSampleDim; ++j) {
		for (long l = 0; l < rest; ++l) {
			pzData[batch * j + l].real(traindata[j][batch * (cnum - 1) + l]);
			pzData[batch * j + l].imag(0);
		}
		for (long l = rest; l < batch; ++l) {
			//pzData[batch * j + l] = 0;
			pzData[batch * j + l].real(0);
			pzData[batch * j + l].imag(0);
		}
	}
	scheme.encrypt(encTrainData[cnum - 1], pzData, slots, wBits, logQ);
	delete[] pzData;
	timeutils.stop("traindata encryption");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Encrypting traindata";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();


	timeutils.start("Encrypting x0:=sumxij...");
	// To Get the sum(Xij) to construct x0
	double sumxij = 0.0;
	for (long i = 0; i < trainSampleDim; ++i)
		for (long j = 0; j < factorDim; ++j)   sumxij += traindata[i][j];
	sumxij = .25*sumxij; // the 1-st B[i][i], namely B[0][0]

	// could encrypt x0 on the client and sent ENC(x0) to the Server !!!
	///////////////////////////////////////////////////////////////////////////// could just keep 0.000 and round up(+0.09) to make  x0 > ...
	double x0 = 2.0 / sumxij  *    .9; // x0 < 2/a, set x0 := 1.8/a
	// if x0 is too close to 2/a, because of the approximate arithmetic of cipher-text, it may lead to a wrong result (negative number).

	Ciphertext* encBinv = new Ciphertext[cnum]; // to store the final result inv(B)
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		// scheme.encryptZeros(encWData[i], slots, wBits, encZData[0].logq); // To Make encVData[0].logq==encZData[0].logq
		scheme.encryptSingle(encBinv[i], x0, wBits+wBits, logQ);
		encBinv[i].n = slots;
	}
	NTL_EXEC_RANGE_END;
	timeutils.stop("Encrypting x0:=sumxij...");         cout << endl << endl;
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Encrypting x0:=sumxij";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

	/* cipherGD.encWVDataZero(encWData, encVData, cnum, slots, wBits, logQ);  */
	timeutils.start("Encrypting weight vector vData...");
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		// scheme.encryptZeros(encWData[i], slots, wBits, encZData[0].logq); // To Make encVData[0].logq==encZData[0].logq
		scheme.encryptSingle(encVData[i], 0.01, wBits, logQ);
		encVData[i].n = slots;
	}
	NTL_EXEC_RANGE_END;
	timeutils.stop("weight vector vData encryption");         cout << endl << endl;
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Encrypting weight vector vData";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();
	/* cipherGD.encWVDataZero(encWData, encVData, cnum, slots, wBits, logQ);  */


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//                                                                                                                                    //
	//                        Client sent (encTrainData, encTrainLabel, enc(x0), encWData, and encVData) to Server                        //
    //                                                                                                                                    //
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////// From now on, the server starts its work on what client sent to it. //////////////////////////////////

	Ciphertext* encZData = new Ciphertext[cnum];
	timeutils.start("encZData = encTrainLabel @ encTrainData ...");
	// To Get the encZData
	NTL_EXEC_RANGE(cnum, first, last);
	for(long i = first; i < last; ++i){
		encZData[i].copy(encTrainLabel[i]);
		scheme.multAndEqual(encZData[i], encTrainData[i]);
		scheme.reScaleByAndEqual(encZData[i], encTrainData[i].logp);
	}
	NTL_EXEC_RANGE_END
	timeutils.stop("encZData is done");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"encZData=encTrainLabel@encTrainData";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();
	/* cipherGD.encZData(encZData, zDataTrain, slots, factorDim, trainSampleDim, batch, cnum, wBits, logQ);  */


	/* --------------------- TEST : encTrainLabel * encTrainData <> encZData --------------------- */
	cout << endl << "encTrainLabel[0] : logp = " << encTrainLabel[0].logp << ", logq = " << encTrainLabel[0].logq << "\t";
	complex<double>* dct1 = scheme.decrypt(secretKey, encTrainLabel[0]);
	for (long l = 0; l < batch; ++l) {
		cout << setiosflags(ios::fixed) << setprecision(10) << dct1[batch * 0 + l].real() << "  ";
	}
	cout << endl << "encTrainData[0]  : logp = " << encTrainData[0].logp <<  ", logq = " << encTrainData[0].logq << "\t";
	complex<double>* dct2 = scheme.decrypt(secretKey, encTrainData[0]);
	for (long l = 0; l < batch; ++l) {
		cout << setiosflags(ios::fixed) << setprecision(10) << dct2[batch * 0 + l].real() << "  ";
	}
	cout << endl;
	cout << endl << "encZData[0]      : logp = " << encZData[0].logp     <<  ", logq = " << encZData[0].logq << "\t";
	complex<double>* dct0 = scheme.decrypt(secretKey, encZData[0]);
	for (long l = 0; l < batch; ++l) {
		cout << setiosflags(ios::fixed) << setprecision(10) << dct0[batch * 0 + l].real() << "  ";
	}
	cout << endl;
	/* --------------------- TEST : encTrainLabel * encTrainData <> encZData --------------------- */


	timeutils.start("Calculating the inverses of B[i][i]");
	/* --------------------- To Calculate the X.T*X (G) --------------------- */
	cout<<"--------------------- To Calculate the X.T*X (G) --------------------- "<<endl<<endl<<endl;
	/* Step 1. Sum Each Row To Its First Element */
	// make a copy of ecnTrainData[i] as encXTX[i]
 	Ciphertext* encXTX = new Ciphertext[cnum];
	for(long i=0; i<cnum; ++i)	encXTX[i].copy(encTrainData[i]);

	/* For Each Batch (ciphertext), Sum Itself Inside */
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		Ciphertext rot;
		for (long l = 0; l < bBits; ++l) {
			scheme.leftRotateFast(rot, encXTX[i], (1 << l));
			scheme.addAndEqual(encXTX[i], rot);
		}
	}
	NTL_EXEC_RANGE_END

	/* Sum All Batchs To Get One Batch */
	Ciphertext encIP;	encIP.copy(encXTX[0]);
	for (long i = 1; i < cnum; ++i) {
		scheme.addAndEqual(encIP, encXTX[i]);
	}

	/* Sum This Batch Inside To Get The ROW SUM */
	scheme.multByPolyNTTAndEqual(encIP, rpoly, pBits, pBits);
	Ciphertext tmp;
	for (long l = 0; l < bBits; ++l) {
		scheme.rightRotateFast(tmp, encIP, (1 << l));
		scheme.addAndEqual(encIP, tmp);
	}
	/* THIS WILL INCREASE logp BY 'pBits', BUT WILL KEEP logq STILL */

	// Now, each row of encIP consists of the same value (sum(X[i][*])


	cout<<" --------------------- Print The Sum of Each Row --------------------- "<<endl;
	complex<double>* dcip = scheme.decrypt(secretKey, encIP);
	for (long j = 0; j < 7; ++j) {
		for (long l = 0; l < batch; ++l) {
			cout << setiosflags(ios::fixed) << setprecision(10) << dcip[batch * j + l].real() << "  ";
		}
		cout << endl;
	}
	cout<<" --------------------- Print The Sum of Each Row --------------------- "<<endl<<endl;


	/* Step 2. Each (i-th) Column (Inner*Product) the result of Step 1. To Get B */
	/*         This step has used/realized the special order of Bonte's method            */

	NTL_EXEC_RANGE(cnum, first, last);
	for(long i=first; i<last; ++i){
		encXTX[i].copy(encTrainData[i]);
		scheme.multAndEqual(encXTX[i], encIP);

		scheme.reScaleByAndEqual(encXTX[i], encIP.logp); // will decrease the logq
		// put off this operation to keep its precision
	}
	NTL_EXEC_RANGE_END

	/* Next : To Sum Each Column To Its First Element, so get every B[i][i] in each row */
	/* It is similar to Sum Each Row To Its First Element, but with a unit move as batch/(1+f) */
	/* For Each Batch, Sum Itself Inside */
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		Ciphertext rot;
		long batch = 1 << bBits;
		for (long l = 0; l < sdimBits; ++l) {
			scheme.leftRotateFast(rot, encXTX[i], (batch << l));
			scheme.addAndEqual(encXTX[i], rot);
		}
	}
	NTL_EXEC_RANGE_END
	/* NOW, EACH ROW OF encXTX[i] HAS EVERY B[i][i] */

	// Now, each column of encXTX[i] consists of the same value B[i][i]

	cout<<" --------------- Print The Sum of Each Column(B[i][i]) --------------- "<<endl;
	complex<double>* dctp = scheme.decrypt(secretKey, encXTX[0]);
	for (long j = 0; j < 7; ++j) {
		for (long l = 0; l < batch; ++l) {
			cout << setiosflags(ios::fixed) << setprecision(10) << dctp[batch * 0 + l].real() << "  ";
		}
		cout << endl;
	}
	cout<<" --------------- Print The Sum of Each Column(B[i][i]) --------------- "<<endl<<endl;

	Ciphertext* encB = new Ciphertext[cnum];
	for (long i = 0; i < cnum; ++i) encB[i].copy(encXTX[i]);
	// Now, each column of encB[i] consists of the same value B[i][i]

	/* Next : DONT forget to * .25 and add the epsion to make it positive */
	double epsion = 1e-8;   epsion *= .25;
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		scheme.multByConstAndEqual(encB[i], .25, pBits);

		scheme.addConstAndEqual(encB[i], epsion, encB[i].logp);

		scheme.reScaleByAndEqual(encB[i], pBits);
	}
	NTL_EXEC_RANGE_END

	cout<<" ------------------- Print .25*B[i][i]+ .25*(1e-8) ------------------- "<<endl;
	complex<double>* dctq = scheme.decrypt(secretKey, encB[0]);
	for (long j = 0; j < 7; ++j) {
		for (long l = 0; l < batch; ++l) {
			cout << setiosflags(ios::fixed) << setprecision(10) << dctq[batch * j + l].real() << "  ";
		}
		cout << endl << setiosflags(ios::fixed) << setprecision(7);
	}
	cout<<" ------------------- Print .25*B[i][i]+ .25*(1e-8) ------------------- "<<endl<<endl;


	/* Step 3. Use Newton Method To Calculate the inv(B) */
	/*         x[k+1] = 2*x[k] - a*x[k]*x[k]             */
	/*                = x[k] * (2 - a*x[k])              */

	// To Get the sum(Xij) to construct x0
	cout<<endl<<".25 * sumxij = "<<sumxij<<endl<<"x0 = 2.0 / sumxij  * .9 = "<<x0<<endl<<endl;

	cout<<" ---------- Use Newton Method To Calculate the inv(B) ---------- "<<endl;
	//Ciphertext* encBinv = new Ciphertext[cnum]; // to store the final result inv(B)

	cout << "before 1-th iteration: encBinv[0].logq = " << encBinv[0].logq <<  ", encBinv[0].logp = " << encBinv[0].logp << endl;
	complex<double>* dctg = scheme.decrypt(secretKey, encBinv[0]);
	for (long j = 0; j < 7; ++j) {
		for (long l = 0; l < batch; ++l) {
			cout << setiosflags(ios::fixed) << setprecision(10)	<< dctg[batch * j + l].real() << "  ";
		}
		cout << endl;
	}
	cout << "before 1-th iteration: encBinv[0].logq = " << encBinv[0].logq <<  ", encBinv[0].logp = " << encBinv[0].logp << endl<<endl;

	long NewtonIter = 9;
	// mod down the initial logq of encBinv[i] for Newton Iteration
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		scheme.modDownToAndEqual(encBinv[i], encB[i].logq);
	}
	NTL_EXEC_RANGE_END;
	
	for ( long it = 0; it < NewtonIter; ++it)
	{
		// (i+1)-th iteration Newton Method For Zero Point (encBinv[i])
		Ciphertext* encTemp = new Ciphertext[cnum];

		NTL_EXEC_RANGE(cnum, first, last);
		for (long i = first; i < last; ++i) {
			encTemp[i].copy(encBinv[i]);

			// square... may avoid the error : x*x<0, do not use mult...
			scheme.squareAndEqual(encTemp[i]);                                     // encTemp = x1 * x1

			scheme.multAndEqual(encTemp[i], encB[i]);                              // encTemp = a * x1 * x1

			scheme.addAndEqual(encBinv[i], encBinv[i]);                            // encBinv[i] = 2 * x1
			                                                                       // avoid to use *, use + instead

            scheme.reScaleByAndEqual(encTemp[i],encTemp[i].logp-encBinv[i].logp);  // MAKE SURE : encBinv[i] and encTemp[i] share
			scheme.modDownToAndEqual(encBinv[i], encTemp[i].logq);                 // the same logp and logq, so they can add or *

			scheme.subAndEqual(encBinv[i], encTemp[i]);                            // encBinv = 2 * x1  - a * x1 * x1

			//scheme.reScaleByAndEqual(encBinv[i], 10);// make encBinv[i].logp equal to the encGrad[i].logp below, but encBinv@encGrad don't need that!
		}
		NTL_EXEC_RANGE_END

		cout << "after "<<(it+1)<<"-th iteration: encBinv[0].logq = " << encBinv[0].logq <<  ", encBinv[0].logp = " << encBinv[0].logp << endl;
		complex<double>* dcth = scheme.decrypt(secretKey, encBinv[0]);
		for (long j = 0; j < 7; ++j) {
			for (long l = 0; l < batch; ++l) {
				cout << setiosflags(ios::fixed) << setprecision(10)	<< dcth[batch * j + l].real() << "  ";
			}
			cout << endl;
		}
		cout << "after "<<(it+1)<<"-th iteration: encBinv[0].logq = " << encBinv[0].logq <<  ", encBinv[0].logp = " << encBinv[0].logp<<endl<<endl;

	}
	/* Each ([i][0~batch)-th) column of encBinv[i] consists of the same value (inv(B[i][i]) */
	timeutils.stop("Calculating the inverses of B[i][i]");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Calculating the inverses of B[i][i]";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();


	/* NTL_EXEC_RANGE and NTL_EXEC_RANGE_END are macros that just do the right thing.
	 * If there are nt threads available, the interval [0..n) will be partitioned into (up to) nt subintervals, and a different thread will be used to process each subinterval. You still have to write the for loop yourself:
	 * the macro just declares and initializes variables first and last (or whatever you want to call them) of type long that represent the subinterval [first..last) to be processed by one thread. */


	double enccor, encauc, truecor, trueauc;


	for (long iter = 0; iter < numIter; ++iter) {
		timeutils.start("BonteSFHwithLearningRate : "+ to_string(iter+1)+" -th iteration");

		cout << endl << endl << endl;
		cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
		cout << "Before the " << iter+1 <<" -th Iteration : encVData[0].logq = " << encVData[0].logq    << endl;
		cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
		cout << endl << endl << endl;


		cout << endl << " !!! START " << iter + 1 << " ITERATION !!! " << endl;
		cout << "--------------------------------------------------------------------------------" << endl << endl;

		cout << "encVData.logq before: " << encVData[0].logq << ",\tencVData.logp before: " << encVData[0].logp << endl;


		/* cipherGD.encNLGDiteration(kdeg, encZData, encWData, encVData, rpoly, cnum, gamma, eta, sBits, bBits, wBits, pBits, aBits); */

			/* CipherGD::encInnerProduct(encIP, encZData, encWData, rpoly, cnum, bBits, wBits, pBits); */
				Ciphertext* encIPvec = new Ciphertext[cnum];

				/* For Each Batch, Sum Itself Inside */
				NTL_EXEC_RANGE(cnum, first, last);
				for (long i = first; i < last; ++i) {
					// MAKE SURE : encZData[i].logq >= encVData[i].logq (and of course : encZData[i].logp == encVData[i].logp)
					if (encZData[i].logq > encVData[i].logq)
						scheme.modDownTo(encIPvec[i], encZData[i], encVData[i].logq); // encIPvec = ENC(zData)
					if (encZData[i].logq < encVData[i].logq)
						scheme.modDownTo(encIPvec[i], encZData[i], encZData[i].logq);
					// V is the final weights to store the result weights.
					scheme.multAndEqual(encIPvec[i], encVData[i]);                // encIPvec = ENC(zData) .* ENC(V)
					//scheme.reScaleByAndEqual(encIPvec[i], encIPvec[i].logp-encVData[i].logp); // maight low down the presice
					//scheme.reScaleByAndEqual(encIPvec[i], pBits);

					/* For Each Batch (==ciphertext), Sum Itself Inside, Result in Each Row consisting of the same value */
					Ciphertext rot;                                               // encIPvec = ENC(zData) @  ENC(V)
					for (long l = 0; l < bBits; ++l) {
						scheme.leftRotateFast(rot, encIPvec[i], (1 << l));
						scheme.addAndEqual(encIPvec[i], rot);
					}
				}
				NTL_EXEC_RANGE_END

				/* Sum All Batchs To Get One Batch */
				Ciphertext encIP; encIP.copy(encIPvec[0]);             // to store the sum of all batches
				for (long i = 1; i < cnum; ++i) {
					scheme.addAndEqual(encIP, encIPvec[i]);
				}
				//scheme.reScaleByAndEqual(encIP, encIP.logp-encVData[0].logp-3*lBits);// could make each row of encIP consist of same value
				//scheme.reScaleByAndEqual(encIP, encIP.logp-wBits);               // could NOT make each row of encIP consist of same value

				/* Sum This Batch Inside To Get The Inner Product */
				scheme.multByPolyNTTAndEqual(encIP, rpoly, pBits, pBits);
				Ciphertext tmp;
				for (long l = 0; l < bBits; ++l) {
					scheme.rightRotateFast(tmp, encIP, (1 << l));
					scheme.addAndEqual(encIP, tmp);
				}
				/* THIS WILL INCREASE logp BY 'pBits', BUT WILL KEEP logq STILL */
				scheme.reScaleByAndEqual(encIP, pBits); // -5 is to make logp equal to encGrad[i].logp

				/* Each (i-th) row of encIP consists of the same value (SumEachRow{encZData[i]@encVData[i]}) */
				/* Each element of encIP's row is the same as  yWTx  in  grad = [Y*(1 - sigm(yWTx))]T * X    */
				delete[] encIPvec;
			/* CipherGD::encInnerProduct(encIP, encZData, encVData, rpoly, cnum, bBits, wBits, pBits); */

			//scheme.reScaleByAndEqual(encIP, pBits);

			cout << endl << "Each row of encIP consists of the same value (yWX), the input of sigmoid()" << endl;
			cout<<endl<<" #-------------------- encIP --------------------# "<<endl;
			cout<<"    encIP.logp  = "<<encIP.logp<<" ,    encIP.logq  = "<<encIP.logq<<endl;
			complex<double>* dcpqj = scheme.decrypt(secretKey, encIP);
			for (long j = 0; j < 12; ++j) {
				for (long l = 0; l < batch; ++l) {
					cout << setiosflags(ios::fixed) << setprecision(10) << dcpqj[batch * j + l].real() << "\t";
				}
				cout << endl;
			}
			cout<<" #-------------------- encIP --------------------# "<<endl;


			/* - - - - - - - - - - - - To Calculate The Gradient - - - - - - - - - - - - */
            /*     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))                         */
            /*           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))                         */
            /*                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))                       */
            /*           grad = [Y*(1 - sigm(yWTx))]T * X	                             */
			/*           grad = (1 - sigm(yWTx)) * Y.T @ X	                             */
			/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
			Ciphertext* encGrad = new Ciphertext[cnum];

			/* CipherGD::encSigmoid(kdeg, encZData, encGrad, encIP, cnum, gamma, sBits, bBits, wBits, aBits); */


				    /* Each (i-th) row of encIP consists of the same value (SumEachRow{encZData[i]@encVData[i]}) */
					Ciphertext encIP2(encIP);
					scheme.multAndEqual(encIP2, encIP);

					// IT IS VERY IMPORT TO KEEP THE logp BIG ENOUGH TO PRESENT THE NUMBER !!  MAKE SURE encIP2.logp>=35
					scheme.reScaleByAndEqual(encIP2, encIP.logp);                // For now, encIP.logp is big enough

					//cout << "eta = " << eta << endl;
					//cout << "gamma = " << gamma << endl;

					////////////////////////////////   y = { -1, +1  }   ///////////////////////////////
					//                                                                                //
					//    gradient = sum(i) : ( 1 - 1/(1 + exp(-yWTX) ) * y * X                       //
					//             = sum(i) : ( 1 - poly(yWTX) ) * y * X                              //
					//             = sum(i) : ( 1 - (0.5 + a*yWTX + b*(yWTX)^3 + ...) ) * y * X       //
					//             = sum(i) : ( 0.5 - a*yWTX - b*(yWTX)^3 - ...) ) * y * X            //
					//                                                                                //
					//                       W[t+1] := W[t] + gamma * gradient                        //
					////////////////////////////////////////////////////////////////////////////////////

					/*--------------# should be 'plus', 'cause to compute the MLE--------------------------------------------*/
					/*--------------# rescale Iter into [-10,+10] using x = (10--10)*(iter-1)/(MaxIter-1) + -10--------------*/
					/*--------------# the learning rate = 2 - y = 2 - 1./(1+e^(-x)) = 1 + 1/(e^x + 1)------------------------*/
				    /*--------------sigmoidx = 20*(iter+1 -1)/(MAX_ITER -1) -10----------------------------------------------*/
				    /*--------------learningrate = 1.0 + 1.0/(exp(sigmoidx) + 1.0)-------------------------------------------*/
					double sigmoidx = 20*(iter+1 -1)/(numIter -1) -10;
					double learningrate = 1.0 + 1.0/(exp(sigmoidx) + 1.0);
					//for (int i = 0; i < factorDim; ++i)	weights[i] = weights[i] + learningrate * B_INV[i]*grad[i];

					if( iter < 5 ){
						//////////////////////////////////////// when iteration < 05 ////////////////////////////////////////
						cout << endl << "INSIDE iter < 5;  poly3 = ";
						cout << setiosflags(ios::showpos) << degree3[0] << " ";
						cout << setiosflags(ios::showpos) << degree3[1] << "x ";
						cout << setiosflags(ios::showpos) << degree3[2] << "x^3 " << endl << endl;
						cout << std::noshowpos;
						// {+0.5,-2.4309e-01, +1.3209e-02};


						scheme.addConstAndEqual(encIP2, degree3[1] / degree3[2], encIP2.logp);                // encIP2 = a/b + yWTx*yWTx

						/* Each (i-th) row of encIP consists of the same value (SumEachRow{encZData[i]@encVData[i]}) */
						/* Each element of encIP's row is the same as  yWTx  in  grad = [Y*(1 - sigm(yWTx))]T * X    */


						NTL_EXEC_RANGE(cnum, first, last);
						//long first = 0, last = cnum;
						for (long i = first; i < last; ++i) {
							//scheme.multAndEqual(encGrad[i], encIP);                                  // encGrad = gamma * Y@X * b * yWTx
							//scheme.reScaleByAndEqual(encGrad[i], pBits);
							/* - - - - - - - - - - - - - - WITH G = inv(B) @ grad - - - - - - - - - - - - - - - - - - - - - - - - */
							// i = 0 : (1+gamma)  * degree3[2] = - 0.0132174;
							// i = 0 : encGrad[0].logp = 0;  encGrad[0].logq = 0;
							scheme.multByConst(encGrad[i], encZData[i], learningrate  * degree3[2], wBits+pBits);
							// i = 0 : encGrad[0].logp = 80; encGrad[0].logq = 983;

							// reScaleToAndEqual influnce logq very much!
							scheme.reScaleByAndEqual(encGrad[i], pBits);                             // encGrad = Y@X *gamma * b


							Ciphertext ctIP(encIP);
							if (encGrad[i].logq > ctIP.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP.logq);     /* whose logq should be ... */
							if (encGrad[i].logq < ctIP.logq)
								scheme.modDownToAndEqual(ctIP, encGrad[i].logq);
							// multiplication doesn't need two ciphertexts.logp to be equal.
							scheme.multAndEqual(encGrad[i], ctIP);                                  // encGrad = gamma * Y@X * b * yWTx
							scheme.reScaleByAndEqual(encGrad[i], ctIP.logp);

							Ciphertext ctIP2(encIP2);
							if(encGrad[i].logq > ctIP2.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP2.logq);
							if(encGrad[i].logq < ctIP2.logq)
								scheme.modDownToAndEqual(ctIP2, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], ctIP2);                                 // encGrad = gamma * Y@X * (a * yWTx + b * yWTx ^3)
							scheme.reScaleByAndEqual(encGrad[i], ctIP2.logp);
							// reScaleByAndEqual & modDownToAndEqual   ! should consider NEXT MOVE!


							Ciphertext tmp;
							// the gamma here should not be (1+gamma) ?
							scheme.multByConst(tmp, encZData[i], learningrate  * degree3[0], wBits);         // tmp = Y@X * gamma * 0.5

							scheme.modDownToAndEqual(tmp, encGrad[i].logq);  // encGrad[i].logq == tmp.logq

							// addition does need two ciphertexts.logp to be equal.// addition also need two ciphertexts.logq to be equal.
							scheme.addAndEqual(encGrad[i], tmp);                                     // encGrad = gamma * Y@X * (0.5 + a * yWTx + b * yWTx ^3)

						}
						NTL_EXEC_RANGE_END;
					/* END OF if(kdeg == 3) {  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

					}else if( iter < 10 ){
						//////////////////////////////////////// when iteration < 10 ////////////////////////////////////////
						cout << endl << "INSIDE iter < 10; poly5 = " ;
						cout << setiosflags(ios::showpos) << degree5[0] << " ";
						cout << setiosflags(ios::showpos) << degree5[1] << "x ";
						cout << setiosflags(ios::showpos) << degree5[2] << "x^3 ";
						cout << setiosflags(ios::showpos) << degree5[3] << "x^5 " <<endl << endl;
						cout << std::noshowpos;


						Ciphertext encIP4;
						scheme.square(encIP4, encIP2);
						// precision is big enough?
						scheme.reScaleByAndEqual(encIP4, encIP2.logp);

						scheme.multByConstAndEqual(encIP2, degree5[2] / degree5[3], wBits);
						scheme.reScaleByAndEqual(encIP2, wBits);

						if(encIP4.logq > encIP2.logq) scheme.modDownToAndEqual(encIP4, encIP2.logq);
						if(encIP4.logq < encIP2.logq) scheme.modDownToAndEqual(encIP2, encIP4.logq);
						scheme.addAndEqual(encIP4, encIP2);

						// encIP4.logp = 60; encIP4.logq = 843;
						// scheme.addConstAndEqual()... need logp to be the same as encIP4.logp
						scheme.addConstAndEqual(encIP4, degree5[1] / degree5[3], encIP4.logp);
						// encIP4.logp = 60; encIP4.logq = 843;

						NTL_EXEC_RANGE(cnum, first, last);
						for (long i = first; i < last; ++i) {
							scheme.multByConst(encGrad[i], encZData[i], learningrate  * degree5[3], wBits+pBits);
							scheme.reScaleByAndEqual(encGrad[i], pBits);

							Ciphertext ctIP(encIP);
							if(encGrad[i].logq > ctIP.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP.logq);
							if(encGrad[i].logq < ctIP.logq)
								scheme.modDownToAndEqual(ctIP, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], ctIP);
							scheme.reScaleByAndEqual(encGrad[i], ctIP.logp);

							Ciphertext ctIP4(encIP4);
							if(encGrad[i].logq > ctIP4.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP4.logq);
							if(encGrad[i].logq < ctIP4.logq)
								scheme.modDownToAndEqual(ctIP4, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], ctIP4);
							scheme.reScaleByAndEqual(encGrad[i], ctIP4.logp);

							Ciphertext tmp;
							scheme.multByConst(tmp, encZData[i], learningrate  * degree5[0], wBits);

							scheme.modDownToAndEqual(tmp, encGrad[i].logq);

							scheme.addAndEqual(encGrad[i], tmp);

						}
						NTL_EXEC_RANGE_END;

					}else{
						//////////////////////////////////////// when iteration < 30 ////////////////////////////////////////
						cout << endl << "INSIDE iter < 30; poly7 = ";
						cout << setiosflags(ios::showpos) << degree7[0] << " ";
						cout << setiosflags(ios::showpos) << degree7[1] << "x " ;
						cout << setiosflags(ios::showpos) << degree7[2] << "x^3 ";
						cout << setiosflags(ios::showpos) << degree7[3] << "x^5 ";
						cout << setiosflags(ios::showpos) << degree7[4] << "x^7 " << endl << endl;
						cout << std::noshowpos;


						if(iter > 30){
							cout << endl << "The Number of Max Iteration should be less than 30!" << endl;
							exit(0);
						}

						Ciphertext encIP4;
						scheme.square(encIP4, encIP2);
						scheme.reScaleByAndEqual(encIP4, encIP2.logp);

						Ciphertext encIP2c;
						scheme.multByConst(encIP2c, encIP2, degree7[3] / degree7[4], wBits);
						scheme.reScaleByAndEqual(encIP2c, wBits);

						if(encIP4.logp != encIP2c.logp) {cout<<"encIP4.logp!=encIP2c.logp"; exit(0); }
						if(encIP4.logq > encIP2c.logq) scheme.modDownToAndEqual(encIP4, encIP2c.logq);
						if(encIP4.logq < encIP2c.logq) scheme.modDownToAndEqual(encIP2c, encIP4.logq);
						scheme.addAndEqual(encIP4, encIP2c);

						//scheme.addConstAndEqual(encIP4, degree7[2] / degree7[4], wBits + 10);
						scheme.addConstAndEqual(encIP4, degree7[2] / degree7[4], encIP4.logp);

						NTL_EXEC_RANGE(cnum, first, last);
						for (long i = first; i < last; ++i) {
							Ciphertext tmp;
							scheme.multByConst(tmp, encZData[i], learningrate  * degree7[1], wBits);

							scheme.modDownToAndEqual(tmp, encIP.logq);

							if(tmp.logq != encIP.logq) {cout << "$$#$$" << endl;exit(0);}

							scheme.multAndEqual(tmp, encIP);
							scheme.reScaleByAndEqual(tmp, encIP.logp);

							//////////////////////////////////////////////////////////////////////////////
							scheme.multByConst(encGrad[i], encZData[i], learningrate  * degree7[0], wBits);
							//scheme.reScaleByAndEqual(encGrad[i], pBits);
							if(tmp.logp > encGrad[i].logp) scheme.reScaleByAndEqual(tmp,tmp.logp-encGrad[i].logp);
							if(tmp.logp < encGrad[i].logp) scheme.reScaleByAndEqual(encGrad[i], encGrad[i].logp-tmp.logp);

							if(tmp.logq > encGrad[i].logq) scheme.modDownToAndEqual(tmp, encGrad[i].logq);
							if(tmp.logq < encGrad[i].logq) scheme.modDownToAndEqual(encGrad[i], tmp.logq);

							scheme.addAndEqual(tmp, encGrad[i]);

							//////////////////////////////////////////////////////////////////////////////
							scheme.multByConst(encGrad[i], encZData[i], learningrate  * degree7[4], wBits + wBits);
							scheme.reScaleByAndEqual(encGrad[i], wBits);

							scheme.modDownToAndEqual(encGrad[i], encIP.logq);

							scheme.multAndEqual(encGrad[i], encIP);

							Ciphertext ctIP2(encIP2);
							if(encGrad[i].logq > ctIP2.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP2.logq);
							if(encGrad[i].logq < ctIP2.logq)
								scheme.modDownToAndEqual(ctIP2, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], ctIP2);
							scheme.reScaleByAndEqual(encGrad[i], ctIP2.logp);

							Ciphertext ctIP4(encIP4);
							if(encGrad[i].logq > ctIP4.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP4.logq);
							if(encGrad[i].logq < ctIP4.logq)
								scheme.modDownToAndEqual(ctIP4, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], encIP4);
							scheme.reScaleByAndEqual(encGrad[i], ctIP4.logp);

							if(tmp.logp > encGrad[i].logp) scheme.reScaleByAndEqual(tmp,tmp.logp-encGrad[i].logp);
							if(tmp.logp < encGrad[i].logp) scheme.reScaleByAndEqual(encGrad[i], encGrad[i].logp-tmp.logp);
							if(tmp.logq > encGrad[i].logq) scheme.modDownToAndEqual(tmp, encGrad[i].logq);
							if(tmp.logq < encGrad[i].logq) scheme.modDownToAndEqual(encGrad[i], tmp.logq);
							scheme.addAndEqual(encGrad[i], tmp);

						}
						NTL_EXEC_RANGE_END;

					}

				// Sum Each Column of encGrad[i] To Get the Final gradient : (1 - sigm(yWTx)) * Y.T @ X
				NTL_EXEC_RANGE(cnum, first, last);
				for (long i = first; i < last; ++i) {
					Ciphertext tmp;
					for (long l = bBits; l < sBits; ++l) {
						scheme.leftRotateFast(tmp, encGrad[i], (1 << l));
						scheme.addAndEqual(encGrad[i], tmp);
					}

					Ciphertext ctBinv(encBinv[i]);
					if (encGrad[i].logq > ctBinv.logq)
						scheme.modDownToAndEqual(encGrad[i], ctBinv.logq);
					if (encGrad[i].logq < ctBinv.logq)
						scheme.modDownToAndEqual(ctBinv, encGrad[i].logq);

					scheme.multAndEqual(encGrad[i], encBinv[i]);
					scheme.reScaleByAndEqual(encGrad[i], encBinv[i].logp);

				}
				NTL_EXEC_RANGE_END;
				/* Each ([i][0~batch)-th) column of encGrad[i] consists of the same value (gamma * encGrad[i][0~batch)) */

				/* In fact, now encGrad has combined with the learning rate gamma */

			/* CipherGD::encSigmoid(kdeg, encZData, encGrad, encIP, cnum, gamma, sBits, bBits, wBits, aBits); */


				cout<<"after (1.+gamma)@encGrad[0], encGrad[0].logp = "<<encGrad[0].logp<<", encGrad[0].logq = "<<encGrad[0].logq<<endl;
				cout<<" $----------- (1.+gamma) @ encGrad[0] -----------$ "<<endl;
				complex<double>* dcpiq2 = scheme.decrypt(secretKey, encGrad[0]);
				for (long j = 0; j < 12; ++j) {
					for (long l = 0; l < batch; ++l) {
						cout << setiosflags(ios::fixed) << setprecision(10) << dcpiq2[batch * j + l].real() << "\t";
					}
					cout << endl << setiosflags(ios::fixed) << setprecision(7);
				}
				cout<<" $----------- (1.+gamma) @ encGrad[0] -----------$ "<<endl<<endl;


			/* CipherGD::encNLGDstep(encWData, encVData, encGrad, eta, cnum, pBits); */
				NTL_EXEC_RANGE(cnum, first, last);
				for (long i = first; i < last; ++i) {
				    // now :: encGrad[i].logp : encVData[i].logp = 60 : 30
				    // now :: encGrad[i].logq : encVData[i].logq = 803 : 983
				    // descrease encGrad[i].logp to equal encVData[i].logp, result in low precision
					if(encGrad[i].logp > encVData[i].logp) scheme.reScaleByAndEqual(encGrad[i], encGrad[i].logp-encVData[i].logp);
					if(encGrad[i].logp < encVData[i].logp) scheme.reScaleByAndEqual(encVData[i], encVData[i].logp-encGrad[i].logp);
					scheme.modDownToAndEqual(encVData[i], encGrad[i].logq);

					scheme.addAndEqual(encVData[i], encGrad[i]); 					// encGrad[i] has already self-multiplied with gamma

				}
				NTL_EXEC_RANGE_END;
	        /* CipherGD::encNLGDstep(encWData, encVData, encGrad, eta, cnum, pBits); */

			delete[] encGrad;


		/* cipherGD.encNLGDiteration(kdeg, encZData, encWData, encVData, rpoly, cnum, gamma, eta, sBits, bBits, wBits, pBits, aBits); */


		cout << "At the end of each iteration : " << endl;
		cout << "\t  logp \t logq " << endl;
		cout << "encGrad[0]: " << encGrad[0].logp << "\t" << encGrad[0].logq << endl;
		cout << "encVData[0]:" << encVData[0].logp << "\t" << encVData[0].logq << endl;
		cout << "encBinv[0]: " << encBinv[0].logp << "\t" << encBinv[0].logq << endl;
		cout << "encZData[0]:" << encZData[0].logp << "\t" << encZData[0].logq << endl;
		cout << "encIP:      " << encIP.logp << "\t" << encIP.logq << endl << endl;

		cout << endl << endl << endl;
		cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
		cout << endl << endl << endl;

		timeutils.stop("BonteSFHwithLearningRate : "+ to_string(iter+1)+" -th iteration");
		openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
		openFileTIMELabel<<","<<"BonteSFHwithLearningRate : "+ to_string(iter+1)+" -th iteration";  openFileTIMELabel.flush();
		openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
		openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

		/////////////////////////////////////////////////////////////////////////////
		//        BOOTSTRAPPING                                                    //
		//             Step 1. Combine various encVData[i] into encVData[0]        //
		//             Step 2. Bootstrap encVData[0]                               //
		//             Step 3. Obtain various encVData[i] from encVData[0]         //
		/////////////////////////////////////////////////////////////////////////////
		if ( encVData[0].logq <= 300 + 90 + pBits + pBits && iter < numIter-1 || encVData[0].logq < wBits && iter == numIter-1) {
			cout << " +-------------------- encVData[0] --------------------+ "	<< endl;
			cout << " encVData[0].logp = " << encVData[0].logp << ", encVData[0].logq = " << encVData[0].logq << endl;
			complex<double>* dcii = scheme.decrypt(secretKey, encVData[0]);
			for (long j = 0; j < 12; ++j) {
				for (long l = 0; l < batch; ++l) {
					cout << setiosflags(ios::fixed) << setprecision(10) << dcii[batch * j + l].real() << "\t";
				}
				cout << endl << setiosflags(ios::fixed) << setprecision(7);
			}
			cout << " +-------------------- encVData[0] --------------------+ "	<< endl ;

			timeutils.start("Use Bootrap To Recrypt Ciphertext");
			cout << endl << " ----------------------- Use Bootrap To Recrypt Ciphertext ----------------------- " << endl;
			// putting encVData and encVData into bootstrap() at the same time may end in error!
			MyMethods::bootstrap(scheme, encVData, cnum, slots, trainSampleDim, batch, logQ);
			//MyMethods::bootstrap(scheme,secretKey, encVData, cnum, slots, trainSampleDim, batch, logQ);
			cout << endl << " ----------------------- Use Bootrap To Recrypt Ciphertext ----------------------- " << endl;
			timeutils.stop("Use Bootrap To Recrypt Ciphertext");

			openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
			openFileTIMELabel<<","<<"Bootstrapping";  openFileTIMELabel.flush();
			openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
			openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

			cout << " x-------------------- encVData[0] --------------------x "	<< endl;
			cout << " encVData[0].logp = " << encVData[0].logp << ", encVData[0].logq = " << encVData[0].logq << endl;
			dcii = scheme.decrypt(secretKey, encVData[0]);
			for (long j = 0; j < 12; ++j) {
				for (long l = 0; l < batch; ++l) {
					cout << setiosflags(ios::fixed) << setprecision(10) << dcii[batch * j + l].real() << "\t";
				}
				cout << endl << setiosflags(ios::fixed) << setprecision(7);
			}
			cout << " x-------------------- encVData[0] --------------------x "	<< endl ;

		}
		/////////////////////////////////////////////////////////////////////////////
		//        BOOTSTRAPPING                                                    //
		//             Over and Out                                                //
		/////////////////////////////////////////////////////////////////////////////


		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cout<<endl<<"---------- TEST : THE "<<iter+1<<"-th ITERATION : Weights, AUC, MLE ----------"<<endl;
		/* cipherGD.decWData(cwData, encWData, factorDim, batch, cnum, wBits);     */
			for (long i = 0; i < (cnum - 1); ++i) {
				complex<double>* dcvv = scheme.decrypt(secretKey, encVData[i]);
				for (long j = 0; j < batch; ++j) {
					cvData[batch * i + j] = dcvv[j].real();
				}
				delete[] dcvv;
			}
			complex<double>* dcvv = scheme.decrypt(secretKey, encVData[cnum-1]);
			long rest = factorDim - batch * (cnum - 1);
			for (long j = 0; j < rest; ++j) {
				cvData[batch * (cnum - 1) + j] = dcvv[j].real();
			}
			delete[] dcvv;
		/* cipherGD.decWData(cwData, encWData, factorDim, batch, cnum, wBits); */
		cout << "Current cWdata (encVData) : " << endl;
		for(long i=0;i<factorDim;++i) cout<<setiosflags(ios::fixed)<<setprecision(12)<<cvData[i]<<",\t";  cout<<endl;

		openFileTestAUC<<","<<MyTools::calculateAUC(zDataTest, cvData, factorDim, testSampleDim, enccor, encauc);    openFileTestAUC.flush();
		openFileTrainAUC<<","<<MyTools::calculateAUC(zDataTrain, cvData, factorDim, trainSampleDim, enccor, encauc); openFileTrainAUC.flush();

		cout << "MLE : " << MyTools::calculateMLE(zDataTest, cvData, factorDim, testSampleDim, enccor, encauc) << endl;
		openFileTestMLE<<","<<MyTools::calculateMLE(zDataTest, cvData, factorDim, testSampleDim, enccor, encauc);    openFileTestMLE.flush();
		openFileTrainMLE<<","<<MyTools::calculateMLE(zDataTrain, cvData, factorDim, trainSampleDim, enccor, encauc); openFileTrainMLE.flush();
		cout << "--------------------------------------------------------------------------------" << endl;
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		cout << " !!! STOP " << iter + 1 << " ITERATION !!! " << endl << endl << endl;
	}

	openFileTIME<<endl;      openFileTIME.flush();
	openFileTIMELabel<<endl; openFileTIMELabel.flush();
	openFileTestAUC<<endl ; openFileTestAUC.flush();
	openFileTrainAUC<<endl ; openFileTrainAUC.flush();
	openFileTestMLE<<endl ; openFileTestMLE.flush();
	openFileTrainMLE<<endl ; openFileTrainMLE.flush();
	openFileCurrMEM<<endl;  openFileCurrMEM.flush();
	openFilePeakMEM<<endl;  openFilePeakMEM.flush();


	openFileTIME.close();
	openFileTIMELabel.close();
	openFileTestAUC.close();
	openFileTrainAUC.close();
	openFileTestMLE.close();
	openFileTrainMLE.close();
	openFileCurrMEM.close();
	openFilePeakMEM.close();

}

double* MyMethods::testCryptoNesterovWithG(double** traindata, double* trainlabel, long factorDim, long trainSampleDim, long numIter, double** testdata, double* testlabel, long testSampleDim, string resultpath)
{
	long fdimBits = (long)ceil(log2(factorDim));              //ceil(x) : Rounds x upward, returning the smallest integral value that is not less than x.
	long sdimBits = (long)ceil(log2(trainSampleDim));         //log2(x) : Returns the binary (base-2) logarithm of x.
/*
	cnum - is number of ciphertexts we need to pack the full dataset.
	wBits - is number of precision bits we have when encrypting the dataset.
	  pBits - is number of precision bits we have for constants encoding (as constants encoding almost have no errors, we can make pBits smaller than wBits)
	  ?lBits - is the number of bits above wBits for the final result (if result is bigger than 1, then we cannot fit it in wBits, we need more)
	aBits - is some parameter for sigmoid evaluation and normally is 2 or 3.
	kdeg - is degree of sigmoid approximation
	slots - number of slots in one ciphertext and is not bigger than N/2 , probably for translating complex to real.
	sBits = log2(slots) from the code
	batch - how many features we pack in one ciphertext.

 */
	long wBits = 30;                                          // Δ (delta)
	long pBits = 20;
	long lBits = 5;
	long aBits = 3;
	long kdeg = 7;
	long kBits = (long)ceil(log2(kdeg));                      // N is the dimension of the Ring; 2^sdimBits is the size of trainSampleDim

	//long MAX_ITER = numIter;	
	//          final result    + iteration * each...
	
	//logQ = 1200;
	cout << "logQ = " << logQ << endl;
	// Do Not use this logQ to bootstrap
	long logN = MyTools::suggestLogN(80, logQ);  // it should be the Security Parameter λ

	long bBits = min(logN - 1 - sdimBits, fdimBits);          // 2^batchBits = min( 2^logN / 2^sdimBits / 2, 2^fdimBits ) = min( N/2 /n, factorDim ) ;
	bBits = 1; 
	if (bBits + sdimBits <= 13) bBits = 13-sdimBits;
	else {
		cout << "WARNING IN THE CHOICE OF bBits OR THE DATASET IS TOO LARGE!" << endl;
		exit(-1);
	}

	// consider the simple way: bBits = min(logN-sdimBits, fdimBits), defined as the number of how many columns in a cipher-text.
	// To be N/2 is because of the translation from complex number to real number.
	long batch = 1 << bBits;         // Basically, batch is the Number of several factor dimensions.
	long sBits = sdimBits + bBits;   // 2^sBits = 2^sdimBits * 2^bBits = ceil2(trainSampleDim) * batch
	long slots =  1 << sBits;        //
	long cnum = (long)ceil((double)factorDim / batch);  // To Divide the whole Train Data into Several Batches (cnum Ciphertexts).

	cout << "batch = " << batch << ", slots = " << slots << ", cnum = " << cnum << endl;
	cout<<"logQ = "<<logQ<<", logN = "<<logN<<", sdimBits = "<<sdimBits<<", fdimBits = "<<fdimBits<<endl;


	//string path = "./data/testCryptoNesterovWithG_";
	string path = resultpath;
	ofstream openFileTrainAUC(path+"TrainAUC.csv",   std::ofstream::out | std::ofstream::app);
	ofstream openFileTrainMLE(path+"TrainMLE.csv",   std::ofstream::out | std::ofstream::app);
	ofstream openFileTestAUC(path+"TestAUC.csv",   std::ofstream::out | std::ofstream::app);
	ofstream openFileTestMLE(path+"TestMLE.csv",   std::ofstream::out | std::ofstream::app);
	ofstream openFileTIME(path+"TIME.csv", std::ofstream::out | std::ofstream::app);
	ofstream openFileTIMELabel(path+"TIMELabel.csv", std::ofstream::out | std::ofstream::app);
	ofstream openFileCurrMEM(path+"CurrMEM.csv", std::ofstream::out | std::ofstream::app);
	ofstream openFilePeakMEM(path+"PeakMEM.csv", std::ofstream::out | std::ofstream::app);

	if(!openFileTrainAUC.is_open()) cout << "Error: cannot read Train AUC file" << endl;
	if(!openFileTrainMLE.is_open()) cout << "Error: cannot read Train MLE file" << endl;
	if(!openFileTestAUC.is_open())  cout << "Error: cannot read Test AUC file" << endl;
	if(!openFileTestMLE.is_open())  cout << "Error: cannot read Test MLE file" << endl;
	if(!openFileTIME.is_open())     cout << "Error: cannot read TIME file" << endl;
	if(!openFileTIMELabel.is_open())cout << "Error: cannot read TIME Label file" << endl;
	if(!openFileTIMELabel.is_open())cout << "Error: cannot read TIME Label file" << endl;
	if(!openFileCurrMEM.is_open())  cout << "Error: cannot read Current MEMORY file" << endl;
	if(!openFilePeakMEM.is_open())  cout << "Error: cannot read Peak MEMORY file" << endl;

	openFileTrainAUC<<"TrainAUC";	openFileTrainAUC.flush();
	openFileTrainMLE<<"TrainMLE";	openFileTrainMLE.flush();
	openFileTestAUC<<"TestAUC";	openFileTestAUC.flush();
	openFileTestMLE<<"TestMLE";	openFileTestMLE.flush();
	openFileTIME<<"TIME";	    openFileTIME.flush();
	openFileTIMELabel<<"TIMELabel";	openFileTIMELabel.flush();
	openFileCurrMEM<<"MEMORY(GB)";	openFileCurrMEM.flush();
	openFilePeakMEM<<"MEMORY(GB)";	openFilePeakMEM.flush();


	TimeUtils timeutils;
	timeutils.start("Scheme generating...");
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);
	scheme.addLeftRotKeys(secretKey);
	scheme.addRightRotKeys(secretKey);
	timeutils.stop("Scheme generation");
	cout << 0 << "-th: TIME= "<< timeutils.timeElapsed << endl;
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Scheme generating";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFilePeakMEM.flush();

	cout << "NOW THE PEAK RSS IS: " << ( MyTools::getPeakRSS() >> 20 ) << endl;
    timeutils.start("Bootstrap Key generating");
    long bootlogq = 40  +10;
    //scheme.addBootKey(secretKey, logn, logq+logI);
    scheme.addBootKey(secretKey, sBits, bootlogq + 4);
    timeutils.stop("Bootstrap Key generated");
    openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
    openFileTIMELabel<<","<<"Bootstrap Key generating";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();
	cout << "NOW THE CURR RSS IS: " << ( MyTools::getCurrentRSS() >> 20 ) << endl;
	cout << "NOW THE PEAK RSS IS: " << ( MyTools::getPeakRSS() >> 20 ) << endl;

	//CipherGD cipherGD(scheme, secretKey);

	// Basically, rpoly is used to calculate the sum{row}
	timeutils.start("Polynomial generating...");
	long np = ceil((pBits + logQ + logN + 2)/59.);
	uint64_t* rpoly = new uint64_t[np << logN];
	/* cipherGD.generateAuxPoly(rpoly, slots, batch, pBits); */
	complex<double>* pvals = new complex<double> [slots];
	for (long j = 0; j < slots; j += batch) {
		pvals[j] = 1.0;
	}
	ZZ* msg = new ZZ[N];
	scheme.ring.encode(msg, pvals, slots, pBits);
	scheme.ring.CRT(rpoly, msg, np);
	delete[] pvals;
	delete[] msg;
	/* cipherGD.generateAuxPoly(rpoly, slots, batch, pBits); */
	timeutils.stop("Polynomial generation");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Polynomial generating";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

	double* cwData = new double[factorDim]; //Current Iteration Weights Data
	double* cvData = new double[factorDim];


	Ciphertext* encTrainData = new Ciphertext[cnum];
	Ciphertext* encTrainLabel= new Ciphertext[cnum];
	Ciphertext* encWData = new Ciphertext[cnum];
	Ciphertext* encVData = new Ciphertext[cnum];


	/* - - - - - - - - - - - - - - - - - - - - - - - - Client and Server - - - - - - - - - - - - - - - - - - - - - - - - */


	// zData = (Y,Y@X)
	double** zDataTrain = new double*[trainSampleDim];
	for(int i=0;i<trainSampleDim;++i)
	{
		double* zi = new double[factorDim];
		zi[0] = trainlabel[i];
		for(int j=1;j<factorDim;++j)
			zi[j] = zi[0]*traindata[i][j];
		zDataTrain[i] = zi;
	}
	// zDataTest is only used for Cross-Validation test, not necessary for training LG model.
	// zData = (Y,Y@X)
	double** zDataTest = new double*[testSampleDim];
	for(int i=0;i<testSampleDim;++i)
	{
		double* zi = new double[factorDim];
		zi[0] = testlabel[i];
		for(int j=1;j<factorDim;++j)
			zi[j] = zi[0]*testdata[i][j];
		zDataTest[i] = zi;
	}

	/* cipherGD.encZData(encZData, zDataTrain, slots, factorDim, trainSampleDim, batch, cnum, wBits, logQ);  */
	timeutils.start("Encrypting trainlabel...");
	// encrypt the trainlabel
	complex<double>* pzLabel = new complex<double>[slots];
	for (long i = 0; i < cnum - 1; ++i) {
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l) {
				pzLabel[batch * j + l].real(trainlabel[j]);
				pzLabel[batch * j + l].imag(0);
			}
		}
		scheme.encrypt(encTrainLabel[i], pzLabel, slots, wBits, logQ);
	}
	long rest = factorDim - batch * (cnum - 1);
	for (long j = 0; j < trainSampleDim; ++j) {
		for (long l = 0; l < rest; ++l) {
			pzLabel[batch * j + l].real(trainlabel[j]);
			pzLabel[batch * j + l].imag(0);
		}
		for (long l = rest; l < batch; ++l) {
			//pzDataLabel[batch * j + l] = 0;
			pzLabel[batch * j + l].real(0);
			pzLabel[batch * j + l].imag(0);
		}
	}
	scheme.encrypt(encTrainLabel[cnum - 1], pzLabel, slots, wBits, logQ);
	delete[] pzLabel;
	timeutils.stop("trainlabel encryption");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Encrypting trainlabel";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

	timeutils.start("Encrypting traindata...");
	// encrypt the traindata
	complex<double>* pzData = new complex<double>[slots];
	for (long i = 0; i < cnum - 1; ++i) {
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l) {
				pzData[batch * j + l].real(traindata[j][batch * i + l]);
				pzData[batch * j + l].imag(0);
			}
		}
		scheme.encrypt(encTrainData[i], pzData, slots, wBits, logQ);
	}
	rest = factorDim - batch * (cnum - 1);
	for (long j = 0; j < trainSampleDim; ++j) {
		for (long l = 0; l < rest; ++l) {
			pzData[batch * j + l].real(traindata[j][batch * (cnum - 1) + l]);
			pzData[batch * j + l].imag(0);
		}
		for (long l = rest; l < batch; ++l) {
			//pzData[batch * j + l] = 0;
			pzData[batch * j + l].real(0);
			pzData[batch * j + l].imag(0);
		}
	}
	scheme.encrypt(encTrainData[cnum - 1], pzData, slots, wBits, logQ);
	delete[] pzData;
	timeutils.stop("traindata encryption");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Encrypting traindata";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();


	timeutils.start("Encrypting x0:=sumxij...");
	// To Get the sum(Xij) to construct x0
	double sumxij = 0.0;
	for (long i = 0; i < trainSampleDim; ++i)
		for (long j = 0; j < factorDim; ++j)   sumxij += traindata[i][j];
	sumxij = .25*sumxij; // the 1-st B[i][i], namely B[0][0]

	// could encrypt x0 on the client and sent ENC(x0) to the Server !!!
	///////////////////////////////////////////////////////////////////////////// could just keep 0.000 and round up(+0.09) to make  x0 > ...
	double x0 = 2.0 / sumxij  *    .9; // x0 < 2/a, set x0 := 1.8/a
	// if x0 is too close to 2/a, because of the approximate arithmetic of cipher-text, it may lead to a wrong result (negative number).

	Ciphertext* encBinv = new Ciphertext[cnum]; // to store the final result inv(B)
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		// scheme.encryptZeros(encWData[i], slots, wBits, encZData[0].logq); // To Make encVData[0].logq==encZData[0].logq
		scheme.encryptSingle(encBinv[i], x0, wBits+wBits, logQ);
		encBinv[i].n = slots;
	}
	NTL_EXEC_RANGE_END;
	timeutils.stop("Encrypting x0:=sumxij...");         cout << endl << endl;
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Encrypting x0:=sumxij";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

	/* cipherGD.encWVDataZero(encWData, encVData, cnum, slots, wBits, logQ);  */
	timeutils.start("Encrypting wData and vData...");
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		// scheme.encryptZeros(encWData[i], slots, wBits, encZData[0].logq); // To Make encVData[0].logq==encZData[0].logq
		scheme.encryptSingle(encWData[i], 0.01, wBits, logQ);
		encWData[i].n = slots;

		encVData[i].copy(encWData[i]);
	}
	NTL_EXEC_RANGE_END;
	timeutils.stop("wData and vData encryption");         cout << endl << endl;
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Encrypting wData and vData";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();
	/* cipherGD.encWVDataZero(encWData, encVData, cnum, slots, wBits, logQ);  */


	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//                                                                                                                                    //
	//                        Client sent (encTrainData, encTrainLabel, enc(x0), encWData, and encVData) to Server                        //
    //                                                                                                                                    //
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////// From now on, the server starts its work on what client sent to it. //////////////////////////////////

	Ciphertext* encZData = new Ciphertext[cnum];
	timeutils.start("encZData = encTrainLabel @ encTrainData ...");
	// To Get the encZData
	NTL_EXEC_RANGE(cnum, first, last);
	for(long i = first; i < last; ++i){
		encZData[i].copy(encTrainLabel[i]);
		scheme.multAndEqual(encZData[i], encTrainData[i]);
		scheme.reScaleByAndEqual(encZData[i], encTrainData[i].logp);
	}
	NTL_EXEC_RANGE_END
	timeutils.stop("encZData is done");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"encZData=encTrainLabel@encTrainData";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();
	/* cipherGD.encZData(encZData, zDataTrain, slots, factorDim, trainSampleDim, batch, cnum, wBits, logQ);  */


	/* --------------------- TEST : encTrainLabel * encTrainData <> encZData --------------------- */
	cout << endl << "encTrainLabel[0] : logp = " << encTrainLabel[0].logp << ", logq = " << encTrainLabel[0].logq << "\t";
	complex<double>* dct1 = scheme.decrypt(secretKey, encTrainLabel[0]);
	for (long l = 0; l < batch; ++l) {
		cout << setiosflags(ios::fixed) << setprecision(10) << dct1[batch * 0 + l].real() << "  ";
	}
	cout << endl << "encTrainData[0]  : logp = " << encTrainData[0].logp <<  ", logq = " << encTrainData[0].logq << "\t";
	complex<double>* dct2 = scheme.decrypt(secretKey, encTrainData[0]);
	for (long l = 0; l < batch; ++l) {
		cout << setiosflags(ios::fixed) << setprecision(10) << dct2[batch * 0 + l].real() << "  ";
	}
	cout << endl;
	cout << endl << "encZData[0]      : logp = " << encZData[0].logp     <<  ", logq = " << encZData[0].logq << "\t";
	complex<double>* dct0 = scheme.decrypt(secretKey, encZData[0]);
	for (long l = 0; l < batch; ++l) {
		cout << setiosflags(ios::fixed) << setprecision(10) << dct0[batch * 0 + l].real() << "  ";
	}
	cout << endl;
	/* --------------------- TEST : encTrainLabel * encTrainData <> encZData --------------------- */


	timeutils.start("Calculating the inverses of B[i][i]");
	/* --------------------- To Calculate the X.T*X (G) --------------------- */
	cout<<"--------------------- To Calculate the X.T*X (G) --------------------- "<<endl<<endl<<endl;
	/* Step 1. Sum Each Row To Its First Element */
	// make a copy of ecnTrainData[i] as encXTX[i]
 	Ciphertext* encXTX = new Ciphertext[cnum];
	for(long i=0; i<cnum; ++i)	encXTX[i].copy(encTrainData[i]);

	/* For Each Batch (ciphertext), Sum Itself Inside */
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		Ciphertext rot;
		for (long l = 0; l < bBits; ++l) {
			scheme.leftRotateFast(rot, encXTX[i], (1 << l));
			scheme.addAndEqual(encXTX[i], rot);
		}
	}
	NTL_EXEC_RANGE_END

	/* Sum All Batchs To Get One Batch */
	Ciphertext encIP;	encIP.copy(encXTX[0]);
	for (long i = 1; i < cnum; ++i) {
		scheme.addAndEqual(encIP, encXTX[i]);
	}

	/* Sum This Batch Inside To Get The ROW SUM */
	scheme.multByPolyNTTAndEqual(encIP, rpoly, pBits, pBits);
	Ciphertext tmp;
	for (long l = 0; l < bBits; ++l) {
		scheme.rightRotateFast(tmp, encIP, (1 << l));
		scheme.addAndEqual(encIP, tmp);
	}
	/* THIS WILL INCREASE logp BY 'pBits', BUT WILL KEEP logq STILL */

	// Now, each row of encIP consists of the same value (sum(X[i][*])


	cout<<" --------------------- Print The Sum of Each Row --------------------- "<<endl;
	complex<double>* dcip = scheme.decrypt(secretKey, encIP);
	for (long j = 0; j < 7; ++j) {
		for (long l = 0; l < batch; ++l) {
			cout << setiosflags(ios::fixed) << setprecision(10) << dcip[batch * j + l].real() << "  ";
		}
		cout << endl;
	}
	cout<<" --------------------- Print The Sum of Each Row --------------------- "<<endl<<endl;


	/* Step 2. Each (i-th) Column (Inner*Product) the result of Step 1. To Get B */
	/*         This step has used/realized the special order of Bonte's method            */

	NTL_EXEC_RANGE(cnum, first, last);
	for(long i=first; i<last; ++i){
		encXTX[i].copy(encTrainData[i]);
		scheme.multAndEqual(encXTX[i], encIP);

		scheme.reScaleByAndEqual(encXTX[i], encIP.logp); // will decrease the logq
		// put off this operation to keep its precision
	}
	NTL_EXEC_RANGE_END

	/* Next : To Sum Each Column To Its First Element, so get every B[i][i] in each row */
	/* It is similar to Sum Each Row To Its First Element, but with a unit move as batch/(1+f) */
	/* For Each Batch, Sum Itself Inside */
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		Ciphertext rot;
		long batch = 1 << bBits;
		for (long l = 0; l < sdimBits; ++l) {
			scheme.leftRotateFast(rot, encXTX[i], (batch << l));
			scheme.addAndEqual(encXTX[i], rot);
		}
	}
	NTL_EXEC_RANGE_END
	/* NOW, EACH ROW OF encXTX[i] HAS EVERY B[i][i] */

	// Now, each column of encXTX[i] consists of the same value B[i][i]

	cout<<" --------------- Print The Sum of Each Column(B[i][i]) --------------- "<<endl;
	complex<double>* dctp = scheme.decrypt(secretKey, encXTX[0]);
	for (long j = 0; j < 7; ++j) {
		for (long l = 0; l < batch; ++l) {
			cout << setiosflags(ios::fixed) << setprecision(10) << dctp[batch * 0 + l].real() << "  ";
		}
		cout << endl;
	}
	cout<<" --------------- Print The Sum of Each Column(B[i][i]) --------------- "<<endl<<endl;

	Ciphertext* encB = new Ciphertext[cnum];
	for (long i = 0; i < cnum; ++i) encB[i].copy(encXTX[i]);
	// Now, each column of encB[i] consists of the same value B[i][i]

	/* Next : DONT forget to * .25 and add the epsion to make it positive */
	double epsion = 1e-8;   epsion *= .25;
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		scheme.multByConstAndEqual(encB[i], .25, pBits);

		scheme.addConstAndEqual(encB[i], epsion, encB[i].logp);

		scheme.reScaleByAndEqual(encB[i], pBits);
	}
	NTL_EXEC_RANGE_END

	cout<<" ------------------- Print .25*B[i][i]+ .25*(1e-8) ------------------- "<<endl;
	complex<double>* dctq = scheme.decrypt(secretKey, encB[0]);
	for (long j = 0; j < 7; ++j) {
		for (long l = 0; l < batch; ++l) {
			cout << setiosflags(ios::fixed) << setprecision(10) << dctq[batch * j + l].real() << "  ";
		}
		cout << endl << setiosflags(ios::fixed) << setprecision(7);
	}
	cout<<" ------------------- Print .25*B[i][i]+ .25*(1e-8) ------------------- "<<endl<<endl;


	/* Step 3. Use Newton Method To Calculate the inv(B) */
	/*         x[k+1] = 2*x[k] - a*x[k]*x[k]             */
	/*                = x[k] * (2 - a*x[k])              */

	// To Get the sum(Xij) to construct x0
	cout<<endl<<".25 * sumxij = "<<sumxij<<endl<<"x0 = 2.0 / sumxij  * .9 = "<<x0<<endl<<endl;

	cout<<" ---------- Use Newton Method To Calculate the inv(B) ---------- "<<endl;
	//Ciphertext* encBinv = new Ciphertext[cnum]; // to store the final result inv(B)

	cout << "before 1-th iteration: encBinv[0].logq = " << encBinv[0].logq <<  ", encBinv[0].logp = " << encBinv[0].logp << endl;
	complex<double>* dctg = scheme.decrypt(secretKey, encBinv[0]);
	for (long j = 0; j < 7; ++j) {
		for (long l = 0; l < batch; ++l) {
			cout << setiosflags(ios::fixed) << setprecision(10)	<< dctg[batch * j + l].real() << "  ";
		}
		cout << endl;
	}
	cout << "before 1-th iteration: encBinv[0].logq = " << encBinv[0].logq <<  ", encBinv[0].logp = " << encBinv[0].logp << endl<<endl;

	long NewtonIter = 9;
	// mod down the initial logq of encBinv[i] for Newton Iteration
	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		scheme.modDownToAndEqual(encBinv[i], encB[i].logq);
	}
	NTL_EXEC_RANGE_END;
	
	for ( long it = 0; it < NewtonIter; ++it)
	{
		// (i+1)-th iteration Newton Method For Zero Point (encBinv[i])
		Ciphertext* encTemp = new Ciphertext[cnum];

		NTL_EXEC_RANGE(cnum, first, last);
		for (long i = first; i < last; ++i) {
			encTemp[i].copy(encBinv[i]);

			// square... may avoid the error : x*x<0, do not use mult...
			scheme.squareAndEqual(encTemp[i]);                                     // encTemp = x1 * x1

			scheme.multAndEqual(encTemp[i], encB[i]);                              // encTemp = a * x1 * x1

			scheme.addAndEqual(encBinv[i], encBinv[i]);                            // encBinv[i] = 2 * x1
			                                                                       // avoid to use *, use + instead

            scheme.reScaleByAndEqual(encTemp[i],encTemp[i].logp-encBinv[i].logp);  // MAKE SURE : encBinv[i] and encTemp[i] share
			scheme.modDownToAndEqual(encBinv[i], encTemp[i].logq);                 // the same logp and logq, so they can add or *

			scheme.subAndEqual(encBinv[i], encTemp[i]);                            // encBinv = 2 * x1  - a * x1 * x1

			//scheme.reScaleByAndEqual(encBinv[i], 10);// make encBinv[i].logp equal to the encGrad[i].logp below, but encBinv@encGrad don't need that!
		}
		NTL_EXEC_RANGE_END

		cout << "after "<<(it+1)<<"-th iteration: encBinv[0].logq = " << encBinv[0].logq <<  ", encBinv[0].logp = " << encBinv[0].logp << endl;
		complex<double>* dcth = scheme.decrypt(secretKey, encBinv[0]);
		for (long j = 0; j < 7; ++j) {
			for (long l = 0; l < batch; ++l) {
				cout << setiosflags(ios::fixed) << setprecision(10)	<< dcth[batch * j + l].real() << "  ";
			}
			cout << endl;
		}
		cout << "after "<<(it+1)<<"-th iteration: encBinv[0].logq = " << encBinv[0].logq <<  ", encBinv[0].logp = " << encBinv[0].logp<<endl<<endl;

	}
	/* Each ([i][0~batch)-th) column of encBinv[i] consists of the same value (inv(B[i][i]) */
	timeutils.stop("Calculating the inverses of B[i][i]");
	openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
	openFileTIMELabel<<","<<"Calculating the inverses of B[i][i]";  openFileTIMELabel.flush();
	openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
	openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();


	/* NTL_EXEC_RANGE and NTL_EXEC_RANGE_END are macros that just do the right thing.
	 * If there are nt threads available, the interval [0..n) will be partitioned into (up to) nt subintervals, and a different thread will be used to process each subinterval. You still have to write the for loop yourself:
	 * the macro just declares and initializes variables first and last (or whatever you want to call them) of type long that represent the subinterval [first..last) to be processed by one thread. */


	double alpha0, alpha1, eta, gamma;
	double enccor, encauc, truecor, trueauc;

	alpha0 = 0.01;
	alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

	for (long iter = 0; iter < numIter; ++iter) {
		timeutils.start("NesterovWithG : "+ to_string(iter+1)+" -th iteration");

		cout << endl << endl << endl;
		cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
		cout << "Before the " << iter+1 <<" -th Iteration : encVData[0].logq = " << encVData[0].logq    << endl;
		cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
		cout << endl << endl << endl;


		cout << endl << " !!! START " << iter + 1 << " ITERATION !!! " << endl;
		cout << "--------------------------------------------------------------------------------" << endl << endl;

		cout << "encVData.logq before: " << encVData[0].logq << ",\tencVData.logp before: " << encVData[0].logp << endl;
		cout << "encWData.logq before: " << encWData[0].logq << ",\tencWData.logp before: " << encWData[0].logp << endl << endl;

		eta = (1 - alpha0) / alpha1;
		double gamma = 1.0 / (iter + 1) / trainSampleDim;


		/* cipherGD.encNLGDiteration(kdeg, encZData, encWData, encVData, rpoly, cnum, gamma, eta, sBits, bBits, wBits, pBits, aBits); */

			/* CipherGD::encInnerProduct(encIP, encZData, encWData, rpoly, cnum, bBits, wBits, pBits); */
				Ciphertext* encIPvec = new Ciphertext[cnum];

				/* For Each Batch, Sum Itself Inside */
				NTL_EXEC_RANGE(cnum, first, last);
				for (long i = first; i < last; ++i) {
					// MAKE SURE : encZData[i].logq >= encVData[i].logq (and of course : encZData[i].logp == encVData[i].logp)
					if (encZData[i].logq > encVData[i].logq)
						scheme.modDownTo(encIPvec[i], encZData[i], encVData[i].logq); // encIPvec = ENC(zData)
					if (encZData[i].logq < encVData[i].logq)
						scheme.modDownTo(encIPvec[i], encZData[i], encZData[i].logq);
					// V is the final weights to store the result weights.
					scheme.multAndEqual(encIPvec[i], encVData[i]);                // encIPvec = ENC(zData) .* ENC(V)
					//scheme.reScaleByAndEqual(encIPvec[i], encIPvec[i].logp-encVData[i].logp); // maight low down the presice
					//scheme.reScaleByAndEqual(encIPvec[i], pBits);

					/* For Each Batch (==ciphertext), Sum Itself Inside, Result in Each Row consisting of the same value */
					Ciphertext rot;                                               // encIPvec = ENC(zData) @  ENC(V)
					for (long l = 0; l < bBits; ++l) {
						scheme.leftRotateFast(rot, encIPvec[i], (1 << l));
						scheme.addAndEqual(encIPvec[i], rot);
					}
				}
				NTL_EXEC_RANGE_END

				/* Sum All Batchs To Get One Batch */
				Ciphertext encIP; encIP.copy(encIPvec[0]);             // to store the sum of all batches
				for (long i = 1; i < cnum; ++i) {
					scheme.addAndEqual(encIP, encIPvec[i]);
				}
				//scheme.reScaleByAndEqual(encIP, encIP.logp-encVData[0].logp-3*lBits);// could make each row of encIP consist of same value
				//scheme.reScaleByAndEqual(encIP, encIP.logp-wBits);               // could NOT make each row of encIP consist of same value

				/* Sum This Batch Inside To Get The Inner Product */
				scheme.multByPolyNTTAndEqual(encIP, rpoly, pBits, pBits);
				Ciphertext tmp;
				for (long l = 0; l < bBits; ++l) {
					scheme.rightRotateFast(tmp, encIP, (1 << l));
					scheme.addAndEqual(encIP, tmp);
				}
				/* THIS WILL INCREASE logp BY 'pBits', BUT WILL KEEP logq STILL */
				scheme.reScaleByAndEqual(encIP, pBits); // -5 is to make logp equal to encGrad[i].logp

				/* Each (i-th) row of encIP consists of the same value (SumEachRow{encZData[i]@encVData[i]}) */
				/* Each element of encIP's row is the same as  yWTx  in  grad = [Y*(1 - sigm(yWTx))]T * X    */
				delete[] encIPvec;
			/* CipherGD::encInnerProduct(encIP, encZData, encVData, rpoly, cnum, bBits, wBits, pBits); */

			//scheme.reScaleByAndEqual(encIP, pBits);

			cout << endl << "Each row of encIP consists of the same value (yWX), the input of sigmoid()" << endl;
			cout<<endl<<" #-------------------- encIP --------------------# "<<endl;
			cout<<"    encIP.logp  = "<<encIP.logp<<" ,    encIP.logq  = "<<encIP.logq<<endl;
			complex<double>* dcpqj = scheme.decrypt(secretKey, encIP);
			for (long j = 0; j < 12; ++j) {
				for (long l = 0; l < batch; ++l) {
					cout << setiosflags(ios::fixed) << setprecision(10) << dcpqj[batch * j + l].real() << "\t";
				}
				cout << endl;
			}
			cout<<" #-------------------- encIP --------------------# "<<endl;


			/* - - - - - - - - - - - - To Calculate The Gradient - - - - - - - - - - - - */
            /*     Note: h(x) = Prob(y=+1|x) = 1/(1+exp(-W.T*x))                         */
            /*           1-h(x)=Prob(y=-1|x) = 1/(1+exp(+W.T*x))                         */
            /*                  Prob(y= I|x) = 1/(1+exp(-I*W.T*x))                       */
            /*           grad = [Y*(1 - sigm(yWTx))]T * X	                             */
			/*           grad = (1 - sigm(yWTx)) * Y.T @ X	                             */
			/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
			Ciphertext* encGrad = new Ciphertext[cnum];

			/* CipherGD::encSigmoid(kdeg, encZData, encGrad, encIP, cnum, gamma, sBits, bBits, wBits, aBits); */


				    /* Each (i-th) row of encIP consists of the same value (SumEachRow{encZData[i]@encVData[i]}) */
					Ciphertext encIP2(encIP);
					scheme.multAndEqual(encIP2, encIP);

					// IT IS VERY IMPORT TO KEEP THE logp BIG ENOUGH TO PRESENT THE NUMBER !!  MAKE SURE encIP2.logp>=35
					scheme.reScaleByAndEqual(encIP2, encIP.logp);                // For now, encIP.logp is big enough

					//cout << "eta = " << eta << endl;
					//cout << "gamma = " << gamma << endl;

					////////////////////////////////   y = { -1, +1  }   ///////////////////////////////
					//                                                                                //
					//    gradient = sum(i) : ( 1 - 1/(1 + exp(-yWTX) ) * y * X                       //
					//             = sum(i) : ( 1 - poly(yWTX) ) * y * X                              //
					//             = sum(i) : ( 1 - (0.5 + a*yWTX + b*(yWTX)^3 + ...) ) * y * X       //
					//             = sum(i) : ( 0.5 - a*yWTX - b*(yWTX)^3 - ...) ) * y * X            //
					//                                                                                //
					//                       W[t+1] := W[t] + gamma * gradient                        //
					////////////////////////////////////////////////////////////////////////////////////

					if( iter < 5 ){
						//////////////////////////////////////// when iteration < 05 ////////////////////////////////////////
						cout << endl << "INSIDE iter < 5;  poly3 = ";
						cout << setiosflags(ios::showpos) << degree3[0] << " ";
						cout << setiosflags(ios::showpos) << degree3[1] << "x ";
						cout << setiosflags(ios::showpos) << degree3[2] << "x^3 " << endl << endl;
						cout << std::noshowpos;
						// {+0.5,-2.4309e-01, +1.3209e-02};


						scheme.addConstAndEqual(encIP2, degree3[1] / degree3[2], encIP2.logp);                // encIP2 = a/b + yWTx*yWTx

						/* Each (i-th) row of encIP consists of the same value (SumEachRow{encZData[i]@encVData[i]}) */
						/* Each element of encIP's row is the same as  yWTx  in  grad = [Y*(1 - sigm(yWTx))]T * X    */


						NTL_EXEC_RANGE(cnum, first, last);
						//long first = 0, last = cnum;
						for (long i = first; i < last; ++i) {
							//scheme.multAndEqual(encGrad[i], encIP);                                  // encGrad = gamma * Y@X * b * yWTx
							//scheme.reScaleByAndEqual(encGrad[i], pBits);
							/* - - - - - - - - - - - - - - WITH G = inv(B) @ grad - - - - - - - - - - - - - - - - - - - - - - - - */
							// i = 0 : (1+gamma)  * degree3[2] = - 0.0132174;
							// i = 0 : encGrad[0].logp = 0;  encGrad[0].logq = 0;
							scheme.multByConst(encGrad[i], encZData[i], (1+gamma)  * degree3[2], wBits+pBits);
							// i = 0 : encGrad[0].logp = 80; encGrad[0].logq = 983;

							// reScaleToAndEqual influnce logq very much!
							scheme.reScaleByAndEqual(encGrad[i], pBits);                             // encGrad = Y@X *gamma * b


							Ciphertext ctIP(encIP);
							if (encGrad[i].logq > ctIP.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP.logq);     /* whose logq should be ... */
							if (encGrad[i].logq < ctIP.logq)
								scheme.modDownToAndEqual(ctIP, encGrad[i].logq);
							// multiplication doesn't need two ciphertexts.logp to be equal.
							scheme.multAndEqual(encGrad[i], ctIP);                                  // encGrad = gamma * Y@X * b * yWTx
							scheme.reScaleByAndEqual(encGrad[i], ctIP.logp);

							Ciphertext ctIP2(encIP2);
							if(encGrad[i].logq > ctIP2.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP2.logq);
							if(encGrad[i].logq < ctIP2.logq)
								scheme.modDownToAndEqual(ctIP2, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], ctIP2);                                 // encGrad = gamma * Y@X * (a * yWTx + b * yWTx ^3)
							scheme.reScaleByAndEqual(encGrad[i], ctIP2.logp);
							// reScaleByAndEqual & modDownToAndEqual   ! should consider NEXT MOVE!


							Ciphertext tmp;
							// the gamma here should not be (1+gamma) ?
							scheme.multByConst(tmp, encZData[i], (1+gamma)  * degree3[0], wBits);         // tmp = Y@X * gamma * 0.5

							scheme.modDownToAndEqual(tmp, encGrad[i].logq);  // encGrad[i].logq == tmp.logq

							// addition does need two ciphertexts.logp to be equal.// addition also need two ciphertexts.logq to be equal.
							scheme.addAndEqual(encGrad[i], tmp);                                     // encGrad = gamma * Y@X * (0.5 + a * yWTx + b * yWTx ^3)

						}
						NTL_EXEC_RANGE_END;
					/* END OF if(kdeg == 3) {  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */

					}else if( iter < 10 ){
						//////////////////////////////////////// when iteration < 10 ////////////////////////////////////////
						cout << endl << "INSIDE iter < 10; poly5 = " ;
						cout << setiosflags(ios::showpos) << degree5[0] << " ";
						cout << setiosflags(ios::showpos) << degree5[1] << "x ";
						cout << setiosflags(ios::showpos) << degree5[2] << "x^3 ";
						cout << setiosflags(ios::showpos) << degree5[3] << "x^5 " <<endl << endl;
						cout << std::noshowpos;


						Ciphertext encIP4;
						scheme.square(encIP4, encIP2);
						// precision is big enough?
						scheme.reScaleByAndEqual(encIP4, encIP2.logp);

						scheme.multByConstAndEqual(encIP2, degree5[2] / degree5[3], wBits);
						scheme.reScaleByAndEqual(encIP2, wBits);

						if(encIP4.logq > encIP2.logq) scheme.modDownToAndEqual(encIP4, encIP2.logq);
						if(encIP4.logq < encIP2.logq) scheme.modDownToAndEqual(encIP2, encIP4.logq);
						scheme.addAndEqual(encIP4, encIP2);

						// encIP4.logp = 60; encIP4.logq = 843;
						// scheme.addConstAndEqual()... need logp to be the same as encIP4.logp
						scheme.addConstAndEqual(encIP4, degree5[1] / degree5[3], encIP4.logp);
						// encIP4.logp = 60; encIP4.logq = 843;

						NTL_EXEC_RANGE(cnum, first, last);
						for (long i = first; i < last; ++i) {
							scheme.multByConst(encGrad[i], encZData[i], (1+gamma)  * degree5[3], wBits+pBits);
							scheme.reScaleByAndEqual(encGrad[i], pBits);

							Ciphertext ctIP(encIP);
							if(encGrad[i].logq > ctIP.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP.logq);
							if(encGrad[i].logq < ctIP.logq)
								scheme.modDownToAndEqual(ctIP, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], ctIP);
							scheme.reScaleByAndEqual(encGrad[i], ctIP.logp);

							Ciphertext ctIP4(encIP4);
							if(encGrad[i].logq > ctIP4.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP4.logq);
							if(encGrad[i].logq < ctIP4.logq)
								scheme.modDownToAndEqual(ctIP4, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], ctIP4);
							scheme.reScaleByAndEqual(encGrad[i], ctIP4.logp);

							Ciphertext tmp;
							scheme.multByConst(tmp, encZData[i], (1+gamma)  * degree5[0], wBits);

							scheme.modDownToAndEqual(tmp, encGrad[i].logq);

							scheme.addAndEqual(encGrad[i], tmp);

						}
						NTL_EXEC_RANGE_END;

					}else{
						//////////////////////////////////////// when iteration < 30 ////////////////////////////////////////
						cout << endl << "INSIDE iter < 30; poly7 = ";
						cout << setiosflags(ios::showpos) << degree7[0] << " ";
						cout << setiosflags(ios::showpos) << degree7[1] << "x " ;
						cout << setiosflags(ios::showpos) << degree7[2] << "x^3 ";
						cout << setiosflags(ios::showpos) << degree7[3] << "x^5 ";
						cout << setiosflags(ios::showpos) << degree7[4] << "x^7 " << endl << endl;
						cout << std::noshowpos;


						if(iter > 30){
							cout << endl << "The Number of Max Iteration should be less than 30!" << endl;
							exit(0);
						}

						Ciphertext encIP4;
						scheme.square(encIP4, encIP2);
						scheme.reScaleByAndEqual(encIP4, encIP2.logp);

						Ciphertext encIP2c;
						scheme.multByConst(encIP2c, encIP2, degree7[3] / degree7[4], wBits);
						scheme.reScaleByAndEqual(encIP2c, wBits);

						if(encIP4.logp != encIP2c.logp) {cout<<"encIP4.logp!=encIP2c.logp"; exit(0); }
						if(encIP4.logq > encIP2c.logq) scheme.modDownToAndEqual(encIP4, encIP2c.logq);
						if(encIP4.logq < encIP2c.logq) scheme.modDownToAndEqual(encIP2c, encIP4.logq);
						scheme.addAndEqual(encIP4, encIP2c);

						//scheme.addConstAndEqual(encIP4, degree7[2] / degree7[4], wBits + 10);
						scheme.addConstAndEqual(encIP4, degree7[2] / degree7[4], encIP4.logp);

						NTL_EXEC_RANGE(cnum, first, last);
						for (long i = first; i < last; ++i) {
							Ciphertext tmp;
							scheme.multByConst(tmp, encZData[i], (1+gamma)  * degree7[1], wBits);

							scheme.modDownToAndEqual(tmp, encIP.logq);

							if(tmp.logq != encIP.logq) {cout << "$$#$$" << endl;exit(0);}

							scheme.multAndEqual(tmp, encIP);
							scheme.reScaleByAndEqual(tmp, encIP.logp);

							//////////////////////////////////////////////////////////////////////////////
							scheme.multByConst(encGrad[i], encZData[i], (1+gamma)  * degree7[0], wBits);
							//scheme.reScaleByAndEqual(encGrad[i], pBits);
							if(tmp.logp > encGrad[i].logp) scheme.reScaleByAndEqual(tmp,tmp.logp-encGrad[i].logp);
							if(tmp.logp < encGrad[i].logp) scheme.reScaleByAndEqual(encGrad[i], encGrad[i].logp-tmp.logp);

							if(tmp.logq > encGrad[i].logq) scheme.modDownToAndEqual(tmp, encGrad[i].logq);
							if(tmp.logq < encGrad[i].logq) scheme.modDownToAndEqual(encGrad[i], tmp.logq);

							scheme.addAndEqual(tmp, encGrad[i]);

							//////////////////////////////////////////////////////////////////////////////
							scheme.multByConst(encGrad[i], encZData[i], (1+gamma)  * degree7[4], wBits + wBits);
							scheme.reScaleByAndEqual(encGrad[i], wBits);

							scheme.modDownToAndEqual(encGrad[i], encIP.logq);

							scheme.multAndEqual(encGrad[i], encIP);

							Ciphertext ctIP2(encIP2);
							if(encGrad[i].logq > ctIP2.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP2.logq);
							if(encGrad[i].logq < ctIP2.logq)
								scheme.modDownToAndEqual(ctIP2, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], ctIP2);
							scheme.reScaleByAndEqual(encGrad[i], ctIP2.logp);

							Ciphertext ctIP4(encIP4);
							if(encGrad[i].logq > ctIP4.logq)
								scheme.modDownToAndEqual(encGrad[i], ctIP4.logq);
							if(encGrad[i].logq < ctIP4.logq)
								scheme.modDownToAndEqual(ctIP4, encGrad[i].logq);
							scheme.multAndEqual(encGrad[i], encIP4);
							scheme.reScaleByAndEqual(encGrad[i], ctIP4.logp);

							if(tmp.logp > encGrad[i].logp) scheme.reScaleByAndEqual(tmp,tmp.logp-encGrad[i].logp);
							if(tmp.logp < encGrad[i].logp) scheme.reScaleByAndEqual(encGrad[i], encGrad[i].logp-tmp.logp);
							if(tmp.logq > encGrad[i].logq) scheme.modDownToAndEqual(tmp, encGrad[i].logq);
							if(tmp.logq < encGrad[i].logq) scheme.modDownToAndEqual(encGrad[i], tmp.logq);
							scheme.addAndEqual(encGrad[i], tmp);

						}
						NTL_EXEC_RANGE_END;

					}

				// Sum Each Column of encGrad[i] To Get the Final gradient : (1 - sigm(yWTx)) * Y.T @ X
				NTL_EXEC_RANGE(cnum, first, last);
				for (long i = first; i < last; ++i) {
					Ciphertext tmp;
					for (long l = bBits; l < sBits; ++l) {
						scheme.leftRotateFast(tmp, encGrad[i], (1 << l));
						scheme.addAndEqual(encGrad[i], tmp);
					}

					Ciphertext ctBinv(encBinv[i]);
					if (encGrad[i].logq > ctBinv.logq)
						scheme.modDownToAndEqual(encGrad[i], ctBinv.logq);
					if (encGrad[i].logq < ctBinv.logq)
						scheme.modDownToAndEqual(ctBinv, encGrad[i].logq);

					scheme.multAndEqual(encGrad[i], encBinv[i]);
					scheme.reScaleByAndEqual(encGrad[i], encBinv[i].logp);
				}
				NTL_EXEC_RANGE_END;
				/* Each ([i][0~batch)-th) column of encGrad[i] consists of the same value (gamma * encGrad[i][0~batch)) */

				/* In fact, now encGrad has combined with the learning rate gamma */

			/* CipherGD::encSigmoid(kdeg, encZData, encGrad, encIP, cnum, gamma, sBits, bBits, wBits, aBits); */


				cout<<"after (1.+gamma)@encGrad[0], encGrad[0].logp = "<<encGrad[0].logp<<", encGrad[0].logq = "<<encGrad[0].logq<<endl;
				cout<<" $----------- (1.+gamma) @ encGrad[0] -----------$ "<<endl;
				complex<double>* dcpiq2 = scheme.decrypt(secretKey, encGrad[0]);
				for (long j = 0; j < 12; ++j) {
					for (long l = 0; l < batch; ++l) {
						cout << setiosflags(ios::fixed) << setprecision(10) << dcpiq2[batch * j + l].real() << "\t";
					}
					cout << endl << setiosflags(ios::fixed) << setprecision(7);
				}
				cout<<" $----------- (1.+gamma) @ encGrad[0] -----------$ "<<endl<<endl;


			/* CipherGD::encNLGDstep(encWData, encVData, encGrad, eta, cnum, pBits); */
				NTL_EXEC_RANGE(cnum, first, last);
				for (long i = first; i < last; ++i) {
				    // now :: encGrad[i].logp : encVData[i].logp = 60 : 30
				    // now :: encGrad[i].logq : encVData[i].logq = 803 : 983
				    // descrease encGrad[i].logp to equal encVData[i].logp, result in low precision
					if(encGrad[i].logp > encVData[i].logp) scheme.reScaleByAndEqual(encGrad[i], encGrad[i].logp-encVData[i].logp);
					if(encGrad[i].logp < encVData[i].logp) scheme.reScaleByAndEqual(encVData[i], encVData[i].logp-encGrad[i].logp);
					scheme.modDownToAndEqual(encVData[i], encGrad[i].logq);

					Ciphertext ctmpw;
					scheme.add(ctmpw, encVData[i], encGrad[i]); 					// encGrad[i] has already self-multiplied with gamma
					                                                                // ctmpw = encVData[i] - encGrad[i]

					scheme.multByConst(encVData[i], ctmpw, 1. - eta, pBits);        // encVData[i] = ( 1. - eta ) * ctmpw
					//scheme.reScaleByAndEqual(encVData[i], pBits-5);


					scheme.multByConstAndEqual(encWData[i], eta, pBits);            // encWData[i] = eta * encWData[i]
					//scheme.reScaleByAndEqual(encWData[i], pBits-5);


					if (encWData[i].logq > encVData[i].logq) scheme.modDownToAndEqual(encWData[i], encVData[i].logq);
					if (encWData[i].logq < encVData[i].logq) scheme.modDownToAndEqual(encVData[i], encWData[i].logq);
					if (encWData[i].logp != encVData[i].logp) { cout << "logp != logp" ;exit(0); }

					scheme.addAndEqual(encVData[i], encWData[i]);                   // encVData[i] = encVData[i] + encWData[i]
					                                                 // encVData[i] = ( 1. - eta ) * ctmpw + eta * encWData[i]

					scheme.reScaleByAndEqual(encVData[i], pBits);
					encWData[i].copy(ctmpw);
				}
				NTL_EXEC_RANGE_END;
	        /* CipherGD::encNLGDstep(encWData, encVData, encGrad, eta, cnum, pBits); */

			delete[] encGrad;

		/* cipherGD.encNLGDiteration(kdeg, encZData, encWData, encVData, rpoly, cnum, gamma, eta, sBits, bBits, wBits, pBits, aBits); */


		cout << "At the end of each iteration : " << endl;
		cout << "\t  logp \t logq " << endl;
		cout << "encGrad[0]: " << encGrad[0].logp << "\t" << encGrad[0].logq << endl;
		cout << "encVData[0]:" << encVData[0].logp << "\t" << encVData[0].logq << endl;
		cout << "encWData[0]:" << encWData[0].logp << "\t" << encWData[0].logq << endl;
		cout << "encBinv[0]: " << encBinv[0].logp << "\t" << encBinv[0].logq << endl;
		cout << "encZData[0]:" << encZData[0].logp << "\t" << encZData[0].logq << endl;
		cout << "encIP:      " << encIP.logp << "\t" << encIP.logq << endl << endl;

		cout << endl << endl << endl;
		cout << "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
		cout << endl << endl << endl;

		timeutils.stop("NesterovWithG : "+ to_string(iter+1)+" -th iteration");
		openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
		openFileTIMELabel<<","<<"NesterovWithG : "+ to_string(iter+1)+" -th iteration";  openFileTIMELabel.flush();
		openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
		openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

		/////////////////////////////////////////////////////////////////////////////
		//        BOOTSTRAPPING                                                    //
		//             Step 1. Combine various encVData[i] into encVData[0]        //
		//             Step 2. Bootstrap encVData[0]                               //
		//             Step 3. Obtain various encVData[i] from encVData[0]         //
		/////////////////////////////////////////////////////////////////////////////
		if ( encVData[0].logq <= 300 + 90 + pBits + pBits && iter < numIter-1 || encVData[0].logq < wBits && iter == numIter-1) {
			cout << " +-------------------- encVData[0] --------------------+ "	<< endl;
			cout << " encVData[0].logp = " << encVData[0].logp << ", encVData[0].logq = " << encVData[0].logq << endl;
			complex<double>* dcii = scheme.decrypt(secretKey, encVData[0]);
			for (long j = 0; j < 12; ++j) {
				for (long l = 0; l < batch; ++l) {
					cout << setiosflags(ios::fixed) << setprecision(10) << dcii[batch * j + l].real() << "\t";
				}
				cout << endl << setiosflags(ios::fixed) << setprecision(7);
			}
			cout << " +-------------------- encVData[0] --------------------+ "	<< endl ;

			timeutils.start("Use Bootrap To Recrypt Ciphertext");
			cout << endl << " ----------------------- Use Bootrap To Recrypt Ciphertext ----------------------- " << endl;
			// putting encVData and encVData into bootstrap() at the same time may end in error!
			MyMethods::bootstrap(scheme, encVData, encWData, cnum, slots, trainSampleDim, batch, logQ);
			//MyMethods::bootstrap(scheme,secretKey, encVData, cnum, slots, trainSampleDim, batch, logQ);
			cout << endl << " ----------------------- Use Bootrap To Recrypt Ciphertext ----------------------- " << endl;
			timeutils.stop("Use Bootrap To Recrypt Ciphertext");
			openFileTIME<<","<<timeutils.timeElapsed;  openFileTIME.flush();
			openFileTIMELabel<<","<<"Bootstrapping";  openFileTIMELabel.flush();
			openFileCurrMEM<<","<< ( MyTools::getCurrentRSS() >> 20 );  openFileCurrMEM.flush();
			openFilePeakMEM<<","<< ( MyTools::getPeakRSS() >> 20 );  openFileCurrMEM.flush();

			cout << " x-------------------- encVData[0] --------------------x "	<< endl;
			cout << " encVData[0].logp = " << encVData[0].logp << ", encVData[0].logq = " << encVData[0].logq << endl;
			dcii = scheme.decrypt(secretKey, encVData[0]);
			for (long j = 0; j < 12; ++j) {
				for (long l = 0; l < batch; ++l) {
					cout << setiosflags(ios::fixed) << setprecision(10) << dcii[batch * j + l].real() << "\t";
				}
				cout << endl << setiosflags(ios::fixed) << setprecision(7);
			}
			cout << " x-------------------- encVData[0] --------------------x "	<< endl ;

		}
		/////////////////////////////////////////////////////////////////////////////
		//        BOOTSTRAPPING                                                    //
		//             Over and Out                                                //
		/////////////////////////////////////////////////////////////////////////////


		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cout<<endl<<"---------- TEST : THE "<<iter+1<<"-th ITERATION : Weights, AUC, MLE ----------"<<endl;
		/* cipherGD.decWData(cwData, encWData, factorDim, batch, cnum, wBits);     */
			for (long i = 0; i < (cnum - 1); ++i) {
				complex<double>* dcvv = scheme.decrypt(secretKey, encVData[i]);
				for (long j = 0; j < batch; ++j) {
					cvData[batch * i + j] = dcvv[j].real();
				}
				delete[] dcvv;
			}
			complex<double>* dcvv = scheme.decrypt(secretKey, encVData[cnum-1]);
			long rest = factorDim - batch * (cnum - 1);
			for (long j = 0; j < rest; ++j) {
				cvData[batch * (cnum - 1) + j] = dcvv[j].real();
			}
			delete[] dcvv;
		/* cipherGD.decWData(cwData, encWData, factorDim, batch, cnum, wBits); */
		cout << "Current cWdata (encVData) : " << endl;
		for(long i=0;i<factorDim;++i) cout<<setiosflags(ios::fixed)<<setprecision(12)<<cvData[i]<<",\t";  cout<<endl;


		openFileTestAUC<<","<<MyTools::calculateAUC(zDataTest, cvData, factorDim, testSampleDim, enccor, encauc);    openFileTestAUC.flush();
		openFileTrainAUC<<","<<MyTools::calculateAUC(zDataTrain, cvData, factorDim, trainSampleDim, enccor, encauc); openFileTrainAUC.flush();

		cout << "MLE : " << MyTools::calculateMLE(zDataTest, cvData, factorDim, testSampleDim, enccor, encauc) << endl;
		openFileTestMLE<<","<<MyTools::calculateMLE(zDataTest, cvData, factorDim, testSampleDim, enccor, encauc);    openFileTestMLE.flush();
		openFileTrainMLE<<","<<MyTools::calculateMLE(zDataTrain, cvData, factorDim, trainSampleDim, enccor, encauc); openFileTrainMLE.flush();
		cout << "--------------------------------------------------------------------------------" << endl;
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


		alpha0 = alpha1;
		alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;
		cout << " !!! STOP " << iter + 1 << " ITERATION !!! " << endl << endl << endl;
	}

	openFileTIME<<endl;      openFileTIME.flush();
	openFileTIMELabel<<endl; openFileTIMELabel.flush();
	openFileTestAUC<<endl ; openFileTestAUC.flush();
	openFileTrainAUC<<endl ; openFileTrainAUC.flush();
	openFileTestMLE<<endl ; openFileTestMLE.flush();
	openFileTrainMLE<<endl ; openFileTrainMLE.flush();
	openFileCurrMEM<<endl;  openFileCurrMEM.flush();
	openFilePeakMEM<<endl;  openFilePeakMEM.flush();

	openFileTIME.close();
	openFileTIMELabel.close();
	openFileTestAUC.close();
	openFileTrainAUC.close();
	openFileTestMLE.close();
	openFileTrainMLE.close();
	openFileCurrMEM.close();
	openFilePeakMEM.close();

}



void MyMethods::bootstrap(Scheme& scheme, Ciphertext* &encVData, Ciphertext* &encWData, long cnum, long slots, long trainSampleDim, long batch, long logQ, long logT, long logI) {

	cout << "cnum, slots, trainSampleDim, batch, logQ, logT, logI = " ;
	cout << cnum << ", " << slots << ", " << trainSampleDim << ", " << batch << ", " << logQ << ", " << logT << ", " << logI << endl;
	logQ = 1200; //// MAKE SURE that the booted logq of cipher is no more than the DEFINE logq that suffice the secret parameter.
                 //// DO NOT NEED the logQ to be the same as the secret parameter.

	/*****************************************************************************************\
	 *                     Combine various encVData[i] into encVData[0]                      *
	 *                so just bootstrapping one ciphertext ecnVData[0] is OK                 *
	\*****************************************************************************************/
	//Ciphertext* encSConst = new Ciphertext[cnum];
	Ciphertext* encSConst = new Ciphertext[2*cnum];
	long pBits = 20;
	if (encVData[0].logq <= 30 + pBits) cout << endl << endl << "SHITS HAPPENED!" << endl << endl;

	// probably a larger scBits or a smaller one will provide a better result
	long scBits = 22; long offset = 10;                //cipher logq after: 353
	//scBits = 22;  offset = 5;

	for (long c = 0; c < cnum; ++c) {
		complex<double>* pcData = new complex<double> [slots];
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l)
				if (j == c)
					pcData[batch * j + l] = 1;
				else
					pcData[batch * j + l] = 0;
		}
		scheme.encrypt(encSConst[c], pcData, slots, scBits, encVData[c].logq);
		delete[] pcData;
	}
	for (long c = cnum; c < 2*cnum; ++c) {
		complex<double>* pcData = new complex<double> [slots];
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l)
				if (j == c)
					pcData[batch * j + l] = 1;
				else
					pcData[batch * j + l] = 0;
		}
		scheme.encrypt(encSConst[c], pcData, slots, scBits, encWData[c-cnum].logq);
		delete[] pcData;
	}
	cout << "Encrypting Special Ciphertext..." ;

	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		scheme.multAndEqual(encVData[i], encSConst[i]); // only need encVData[0] to be done
		//scheme.reScaleByAndEqual(encVData[i], scBits-offset);   // delay&save this operations to bootstrapping
	}
	NTL_EXEC_RANGE_END;
	cout << "Special Ciphertext @ encVData ..." ;

	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		scheme.multAndEqual(encWData[i], encSConst[i+cnum]); // only need encVData[0] to be done
		//scheme.reScaleByAndEqual(encWData[i], scBits-offset);   // delay&save this operations to bootstrapping
	}
	NTL_EXEC_RANGE_END;
	cout << "Special Ciphertext @ encWData ..." << endl;


	// MAKE SURE : cnum < 2^sdimBits !!!
	for (long l = 1; l < cnum; ++l) {
		// adding a RotKey or a BootKey twice may lead to a error!
		scheme.addAndEqual(encVData[0], encVData[l]);
	}
	cout << "Combine encVData[i] into encVData[0] ..." ;

	for (long l = 0; l < cnum; ++l) {
		// adding a RotKey or a BootKey twice may lead to a error!
		scheme.addAndEqual(encVData[0], encWData[l]);
	}
	cout << "Combine encWData[i] into encVData[0] ..." << endl ;

	scheme.reScaleByAndEqual(encVData[0], scBits-offset);
	/*****************************************************************************************\
	 *                      Combine various encVData[i] into encVData[0]                     *
	 *                Now, encVData[0] contains all the information you want !               *
	\*****************************************************************************************/


	cout << " ------------------------ BOOTSTRAPPING BEGINNING ------------------------ " << endl;

	        long logp = 40; logp = encVData[0].logp;
	        cout << " bootstrap.logp = " << logp << " = encVData[0].logp" << endl;

	        long logq = logp + 10; //< suppose the input ciphertext of bootstrapping has logq = logp + 10
	        //long logSlots = sBits; //( = 12); //< larger logn will make bootstrapping tech much slower
	        long logSlots = log2(encVData[0].n);
	        //long logT = 3; //< this means that we use Taylor approximation in [-1/T,1/T] with double a    ngle fomula
	        // It works that logT is 3 and logp is 35 and logq = logp + 10; logT being 4 result in error!
	        // logT being 3 works better than any other number!
	        //TestScheme::testBootstrap(logq, logp, logn, logT);
	        //void TestScheme::testBootstrap(long logq, long logp, long logSlots, long logT) {
			cout << "!!! START TEST BOOTSTRAP !!!" << endl;

	        Ciphertext cipher(encVData[0]);
	        ///////////////// MAKE the logp and logq of encVData[0] EQUAL TO logp and loq /////////////////
	        if (cipher.logp > logp) scheme.reScaleToAndEqual(cipher, logp);
	        if (cipher.logp < logp) { cout << "ERROR!" << endl; exit(0); }
	        if (cipher.logq > logq) scheme.modDownToAndEqual(cipher, logq);
	        if (cipher.logq < logq) { cout << "ERROR@" << endl; exit(0); }

	        cout << "cipher logq before: " << cipher.logq << endl;

	        scheme.modDownToAndEqual(cipher, logq);
	        scheme.normalizeAndEqual(cipher);
	        cipher.logq = logQ;
	        cipher.logp = logq + 4;

	        Ciphertext rott;
	        for (long i = logSlots; i < logNh; ++i) {
	            scheme.leftRotateFast(rott, cipher, (1 << i));
	            scheme.addAndEqual(cipher, rott);
	        }
	        scheme.divByPo2AndEqual(cipher, logNh);
	        cout << "SubSum\t" ;

	        scheme.coeffToSlotAndEqual(cipher);
	        cout << "CoeffToSlot\t " ;

	        scheme.evalExpAndEqual(cipher, logT);
	        cout << "EvalExp\t ";

	        scheme.slotToCoeffAndEqual(cipher);
	        cout << "SlotToCoeff" << endl;

	        cipher.logp = logp;

	        cout << "cipher logq after: " << cipher.logq << endl;


	cout << " ------------------------ BOOTSTRAPPING FINISHED ------------------------ " << endl;


	/*****************************************************************************************\
	 *                      Get various encVData[i] from encVData[0]                         *
	 *             a new fresh ciphertext ecnVData[0] with bigger modulo is ready            *
	\*****************************************************************************************/

	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i)
		encVData[i].copy(cipher);
	NTL_EXEC_RANGE_END;

	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i)
		encWData[i].copy(cipher);
	NTL_EXEC_RANGE_END;

	//// construct the special constant ciphertext
	// MUST Encrypt the Special Ciphertext AGAIN! SO THAT the logq of encVData[c] and encSConst[c] can be equal!
	// may be can construct encSConst[c] with the largest logq, copy it and mod down to the needed logq when use it
	for (long c = 0; c < cnum; ++c) {
		complex<double>* pcData = new complex<double> [slots];
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l)
				if (j == c)
					pcData[batch * j + l] = 1;
				else
					pcData[batch * j + l] = 0;
		}
		scheme.encrypt(encSConst[c], pcData, slots, scBits, encVData[c].logq);
		delete[] pcData;
	}
	for (long c = cnum; c < 2*cnum; ++c) {
		complex<double>* pcData = new complex<double> [slots];
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l)
				if (j == c)
					pcData[batch * j + l] = 1;
				else
					pcData[batch * j + l] = 0;
		}
		scheme.encrypt(encSConst[c], pcData, slots, scBits, encWData[c-cnum].logq);
		delete[] pcData;
	}
	cout << "Encrypting Special Ciphertext..." ;

	NTL_EXEC_RANGE(cnum, first, last);
	for ( long c = first; c < last; ++c) {
		scheme.multAndEqual(encVData[c], encSConst[c]);
		// if still use the encVData[i] in-place, should watch out its logp and logq
		//scheme.reScaleByAndEqual(encVData[c], scBits + offset);
	}
	NTL_EXEC_RANGE_END;
	cout << "Special Ciphertext @ encVData ..." ;


	NTL_EXEC_RANGE(cnum, first, last);
	for ( long c = first; c < last; ++c) {
		scheme.multAndEqual(encWData[c], encSConst[c+cnum]);
		//scheme.reScaleByAndEqual(encWData[c], scBits + offset);
	}
	NTL_EXEC_RANGE_END;
	cout << "Special Ciphertext @ encWData ..." << endl;

	long sdimBits = (long)ceil(log2(trainSampleDim));
	//cout << "sdimBits = " << sdimBits << endl;
	//cout << "batch = " << batch << endl;
	NTL_EXEC_RANGE(cnum, first, last);
	for (long c = first; c < last; ++c) {
		Ciphertext rot;
		//long batch = 1 << bBits;
		for (long l = 0; l < sdimBits; ++l) {
			scheme.leftRotateFast(rot, encVData[c], (batch << l));
			scheme.addAndEqual(encVData[c], rot);
		}
	}
	NTL_EXEC_RANGE_END
	cout << "Got encVData[i] from encVData[0] ..." ;

	NTL_EXEC_RANGE(cnum, first, last);
	for (long c = first; c < last; ++c) {
		Ciphertext rot;
		//long batch = 1 << bBits;
		for (long l = 0; l < sdimBits; ++l) {
			scheme.leftRotateFast(rot, encWData[c], (batch << l));
			scheme.addAndEqual(encWData[c], rot);
		}
	}
	NTL_EXEC_RANGE_END

	NTL_EXEC_RANGE(cnum, first, last);
	for ( long c = first; c < last; ++c) {
		scheme.reScaleByAndEqual(encVData[c], scBits + offset);
		scheme.reScaleByAndEqual(encWData[c], scBits + offset);
	}
	NTL_EXEC_RANGE_END;

	cout << "Got encWData[i] from encWData[0] ..." << endl;
	delete[] encSConst;
	/*****************************************************************************************\
	 *                       Get various encVData[i] from encVData[0]                        *
	 *                                      It is Done!                                      *
	\*****************************************************************************************/
}


void MyMethods::bootstrap(Scheme& scheme, Ciphertext* &encVData, long cnum, long slots, long trainSampleDim, long batch, long logQ, long logT, long logI) {

	cout << "cnum, slots, trainSampleDim, batch, logQ, logT, logI = " ;
	cout << cnum << ", " << slots << ", " << trainSampleDim << ", " << batch << ", " << logQ << ", " << logT << ", " << logI << endl;
	logQ = 1200; //// MAKE SURE that the booted logq of cipher is no more than the DEFINE logq that suffice the secret parameter.
                 //// DO NOT NEED the logQ to be the same as the secret parameter.

	/*****************************************************************************************\
	 *                     Combine various encVData[i] into encVData[0]                      *
	 *                so just bootstrapping one ciphertext ecnVData[0] is OK                 *
	\*****************************************************************************************/
	//Ciphertext* encSConst = new Ciphertext[cnum];
	Ciphertext* encSConst = new Ciphertext[2*cnum];
	long pBits = 20;
	if (encVData[0].logq <= 30 + pBits) cout << endl << endl << "SHITS HAPPENED!" << endl << endl;

	// probably a larger scBits or a smaller one will provide a better result
	long scBits = 22; long offset = 10;                //cipher logq after: 353
	//scBits = 22;  offset = 5;

	for (long c = 0; c < cnum; ++c) {
		complex<double>* pcData = new complex<double> [slots];
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l)
				if (j == c)
					pcData[batch * j + l] = 1;
				else
					pcData[batch * j + l] = 0;
		}
		scheme.encrypt(encSConst[c], pcData, slots, scBits, encVData[c].logq);
		delete[] pcData;
	}
	cout << "Encrypting Special Ciphertext..." ;

	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i) {
		scheme.multAndEqual(encVData[i], encSConst[i]); // only need encVData[0] to be done
		//scheme.reScaleByAndEqual(encVData[i], scBits-offset);   // delay&save this operations to bootstrapping
	}
	NTL_EXEC_RANGE_END;
	cout << "Special Ciphertext @ encVData ..." ;


	// MAKE SURE : cnum < 2^sdimBits !!!
	for (long l = 1; l < cnum; ++l) {
		// adding a RotKey or a BootKey twice may lead to a error!
		scheme.addAndEqual(encVData[0], encVData[l]);
	}
	cout << "Combine encVData[i] into encVData[0] ..." << endl;


	scheme.reScaleByAndEqual(encVData[0], scBits-offset);
	/*****************************************************************************************\
	 *                      Combine various encVData[i] into encVData[0]                     *
	 *                Now, encVData[0] contains all the information you want !               *
	\*****************************************************************************************/


	cout << " ------------------------ BOOTSTRAPPING BEGINNING ------------------------ " << endl;

	        long logp = 40; logp = encVData[0].logp;
	        cout << " bootstrap.logp = " << logp << " = encVData[0].logp" << endl;

	        long logq = logp + 10; //< suppose the input ciphertext of bootstrapping has logq = logp + 10
	        //long logSlots = sBits; //( = 12); //< larger logn will make bootstrapping tech much slower
	        long logSlots = log2(encVData[0].n);
	        //long logT = 3; //< this means that we use Taylor approximation in [-1/T,1/T] with double a    ngle fomula
	        // It works that logT is 3 and logp is 35 and logq = logp + 10; logT being 4 result in error!
	        // logT being 3 works better than any other number!
	        //TestScheme::testBootstrap(logq, logp, logn, logT);
	        //void TestScheme::testBootstrap(long logq, long logp, long logSlots, long logT) {
			cout << "!!! START TEST BOOTSTRAP !!!" << endl;

	        Ciphertext cipher(encVData[0]);
	        ///////////////// MAKE the logp and logq of encVData[0] EQUAL TO logp and loq /////////////////
	        if (cipher.logp > logp) scheme.reScaleToAndEqual(cipher, logp);
	        if (cipher.logp < logp) { cout << "ERROR!" << endl; exit(0); }
	        if (cipher.logq > logq) scheme.modDownToAndEqual(cipher, logq);
	        if (cipher.logq < logq) { cout << "ERROR@" << endl; exit(0); }

	        cout << "cipher logq before: " << cipher.logq << endl;

	        scheme.modDownToAndEqual(cipher, logq);
	        scheme.normalizeAndEqual(cipher);
	        cipher.logq = logQ;
	        cipher.logp = logq + 4;

	        Ciphertext rott;
	        for (long i = logSlots; i < logNh; ++i) {
	            scheme.leftRotateFast(rott, cipher, (1 << i));
	            scheme.addAndEqual(cipher, rott);
	        }
	        scheme.divByPo2AndEqual(cipher, logNh);
	        cout << "SubSum\t" ;

	        scheme.coeffToSlotAndEqual(cipher);
	        cout << "CoeffToSlot\t " ;

	        scheme.evalExpAndEqual(cipher, logT);
	        cout << "EvalExp\t ";

	        scheme.slotToCoeffAndEqual(cipher);
	        cout << "SlotToCoeff" << endl;

	        cipher.logp = logp;

	        cout << "cipher logq after: " << cipher.logq << endl;


	cout << " ------------------------ BOOTSTRAPPING FINISHED ------------------------ " << endl;


	/*****************************************************************************************\
	 *                      Get various encVData[i] from encVData[0]                         *
	 *             a new fresh ciphertext ecnVData[0] with bigger modulo is ready            *
	\*****************************************************************************************/

	NTL_EXEC_RANGE(cnum, first, last);
	for (long i = first; i < last; ++i)
		encVData[i].copy(cipher);
	NTL_EXEC_RANGE_END;

	//// construct the special constant ciphertext
	// MUST Encrypt the Special Ciphertext AGAIN! SO THAT the logq of encVData[c] and encSConst[c] can be equal!
	// may be can construct encSConst[c] with the largest logq, copy it and mod down to the needed logq when use it
	for (long c = 0; c < cnum; ++c) {
		complex<double>* pcData = new complex<double> [slots];
		for (long j = 0; j < trainSampleDim; ++j) {
			for (long l = 0; l < batch; ++l)
				if (j == c)
					pcData[batch * j + l] = 1;
				else
					pcData[batch * j + l] = 0;
		}
		scheme.encrypt(encSConst[c], pcData, slots, scBits, encVData[c].logq);
		delete[] pcData;
	}
	cout << "Encrypting Special Ciphertext..." ;

	NTL_EXEC_RANGE(cnum, first, last);
	for ( long c = first; c < last; ++c) {
		scheme.multAndEqual(encVData[c], encSConst[c]);
		// if still use the encVData[i] in-place, should watch out its logp and logq
		//scheme.reScaleByAndEqual(encVData[c], scBits + offset);
	}
	NTL_EXEC_RANGE_END;
	cout << "Special Ciphertext @ encVData ..." ;

	long sdimBits = (long)ceil(log2(trainSampleDim));
	//cout << "sdimBits = " << sdimBits << endl;
	//cout << "batch = " << batch << endl;
	NTL_EXEC_RANGE(cnum, first, last);
	for (long c = first; c < last; ++c) {
		Ciphertext rot;
		//long batch = 1 << bBits;
		for (long l = 0; l < sdimBits; ++l) {
			scheme.leftRotateFast(rot, encVData[c], (batch << l));
			scheme.addAndEqual(encVData[c], rot);
		}
	}
	NTL_EXEC_RANGE_END


	NTL_EXEC_RANGE(cnum, first, last);
	for ( long c = first; c < last; ++c) {
		scheme.reScaleByAndEqual(encVData[c], scBits + offset);
	}
	NTL_EXEC_RANGE_END;
	cout << "Got encVData[i] from encVData[0] ..." ;
	delete[] encSConst;
	/*****************************************************************************************\
	 *                       Get various encVData[i] from encVData[0]                        *
	 *                                      It is Done!                                      *
	\*****************************************************************************************/
}

