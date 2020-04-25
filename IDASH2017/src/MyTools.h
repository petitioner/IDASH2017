/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

////////////////////////////////////////////////////////////////////////////////////
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif
////////////////////////////////////////////////////////////////////////////////////

#ifndef IDASH2017_GD_H_
#define IDASH2017_GD_H_

#include <iostream>

////////////////////////////////   y = { -1, +1  }   ///////////////////////////////
//                                                                                //
//    gradient = sum(i) : ( 1 - 1/(1 + exp(-yWTX) ) * y * X                       //
//             = sum(i) : ( 1 - poly(yWTX) ) * y * X                              //
//             = sum(i) : ( 1 - (0.5 + a*yWTX + b*(yWTX)^3 + ...) ) * y * X       //
//             = sum(i) : ( 0.5 - a*yWTX - b*(yWTX)^3 - ...) ) * y * X            //
//                                                                                //
//                       W[t+1] := W[t] + gamma * gradient                        //
////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////// JUST FOR IDASH2017 ////////////////////////////////
//             Round 1  ~ 05 : |wTx| < ?????                                      //
//             Round 05 ~ 10 : |wTx| < ?????                                      //
//             Round 10 ~ 30 : |wTx| < ?????                                      //
////////////////////////////////////////////////////////////////////////////////////
//static double degree1[2] = {0.5,0.25};
// y = 0.5  +  0.15012 x^1  -  0.0015930078125 x^3                 polyfit(x,y,3)
static double degree3[3] = {+0.5, -0.15012, +0.0015930078125};  // 1 - poly(-yWTx)
// y = 0.5  +  0.19131 x^1  -  0.0045963 x^3         +  4.123320007324219e-05 x^5  
static double degree5[4] = {+0.5, -0.19131, +0.0045963,  -4.123320007324219e-05};
// y = 0.5  +  0.21687 x^1  -  0.00819154296875 x^3  +  0.0001658331298828125 x^5  -  1.1956167221069336e-06 x^7
static double degree7[5] = {+0.5, -0.21687, +0.00819154296875, -0.0001658331298828125, +1.1956167221069336e-06};
//////////////////////////////// JUST FOR IDASH2017 ////////////////////////////////

//////////////////////////////// JUST FOR MNIST train //////////////////////////////
//             Round 1  ~ 05 : |wTx| < 4                                          //
//             Round 05 ~ 10 : |wTx| < 6                                          //
//             Round 10 ~ 30 : |wTx| < 12                                         //
////////////////////////////////////////////////////////////////////////////////////
//static double degree1[2] = {0.5,0.25};
// y = 0.5  +  0.15012 x^1  -  0.0015930078125 x^3
//static double degree3[3] = {+0.5, -0.15012, +0.0015930078125};  // 1 - poly(-yWTx)
// y = 0.5  +  0.19131 x^1  -  0.0045963 x^3         +  4.123320007324219e-05 x^5  
//static double degree5[4] = {+0.5, -0.19131, +0.0045963,  -4.123320007324219e-05};
// y = 0.5  +  0.14465 x^1  -  0.0024308 x^3         +  0.000021893 x^5  -  0.000000070222 x^7
//static double degree7[5] = {+0.5, -0.14465, +0.0024308, -0.000021893, +0.000000070222};
//////////////////////////////// JUST FOR MNIST train //////////////////////////////


using namespace std;

class MyTools {

public:

	static long suggestLogN(long lambda, long logQ);

	static double** dataFromFile(string& path, long& factorDim, long& sampleDim, double** &X, double* &Y);
	static double** zDataFromFile(string& path, long& factorDim, long& sampleDim, bool isfirst = true);

	static void shuffleDataSync(double** X, long factorDim, long sampleDim, double* Y);
	static void shuffleZData(double** zData, long factorDim, long sampleDim);

	static void normalizeZData(double** zData, long factorDim, long sampleDim);
	static void normalizezData2(double** zDataLearn, double** zDataTest, long factorDim, long sampleDimLearn, long sampleDimTest);

	static void initialWDataVDataAverage(double* wData, double* vData, double** zData, long factorDim, long sampleDim);
	static void initialWDataVDataZero(double* wData, double* vData, long factorDim);

	static double* plainIP(double** a, double* b, long factorDim, long sampleDim);
	static double* plainSigmoid(long approxDeg, double** zData, double* ip, long factorDim, long sampleDim, double gamma);
	/* add by John L. Smith */
	static double* plainSigmoid(long approxDeg, double** zData, double* ip, long factorDim, long sampleDim);

	static void plainLGDstep(double* wData, double* grad, long factorDim);
	static void plainMLGDstep(double* wData, double* vData, double* grad, long factorDim, double eta);
	static void plainNLGDstep(double* wData, double* vData, double* grad, long factorDim, double eta);
	// added by John L. Smith
	static void plainNLGDstep(double* wData, double* vData, double* grad, long factorDim, double eta, double gamma);

	static void plainLGDL2step(double* wData, double* grad, long factorDim, double lambda);
	static void plainMLGDL2step(double* wData, double* vData, double* grad, long factorDim, double eta, double lambda);
	static void plainNLGDL2step(double* wData, double* vData, double* grad, long factorDim, double eta, double lambda);

	static void plainLGDiteration(long approxDeg, double** zData, double* wData, long factorDim, long sampleDim, double gamma);
	static void plainMLGDiteration(long approxDeg, double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta);
	static void plainNLGDiteration(long approxDeg, double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta);
	// added by John L. Smith
	static void plainNesterovWithGbyXTXiteration(long approxDeg, double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta, double * G);

	static void plainLGDL2iteration(long approxDeg, double** zData, double* wData, long factorDim, long sampleDim, double gamma, double lambda);
	static void plainMLGDL2iteration(long approxDeg, double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta, double lambda);
	static void plainNLGDL2iteration(long approxDeg, double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta, double lambda);

	//-----------------------------------------

	static double trueIP(double* a, double* b, long size);

	static void trueLGDiteration(double** zData, double* wData, long factorDim, long sampleDim, double gamma);
	static void trueMLGDiteration(double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta);
	static void trueNLGDiteration(double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta);
	// added by John L. Smith
	static void trueNesterovWithGbyXTXiteration(double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta, double * G);

	static void trueLGDL2iteration(double** zData, double* wData, long factorDim, long sampleDim, double gamma, double lambda);
	static void trueMLGDL2iteration(double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta, double lambda);
	static void trueNLGDL2iteration(double** zData, double* wData, double* vData, long factorDim, long sampleDim, double gamma, double eta, double lambda);

	static double calculateAUC(double** zData, double* wData, long factorDim, long sampleDim, double& correctness, double& AUC);
	static double calculateMLE(double** zData, double* wData, long factorDim, long sampleDim, double& correctness, double& auc);
	static double calculateMSE(double* wData1, double* wData2, long factorDim);
	static double calculateNMSE(double* wData1, double* wData2, long factorDim);

	////////////////////////////////////////////
	static size_t getPeakRSS( );
	static size_t getCurrentRSS( );

};

#endif /* SGD_SGD_H_ */
