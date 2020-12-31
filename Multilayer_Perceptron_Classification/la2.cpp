//============================================================================
// Introducción a los Modelos Computacionales
// Name        : practica1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h> // For DBL_MAX

#include "imc/MultilayerPerceptron.h"


using namespace imc;
using namespace std;

int main(int argc, char **argv) {
	// Process the command line
	bool wflag = 0, pflag = 0, Tflag= 0;
	char *Tvalue = NULL, *wvalue = NULL;
	char *trainingFile = NULL, *testingFile = NULL;
	int c, iterations = 1000, hiddenLayers = 1, neuronsHiddenLayers = 5, functionUsed = 0, softmax = 0;
	double eta = 0.1, mu = 0.9, validationRatio = 0.0, decreasingFactor = 1;
	bool online = false;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:v:d:w:f:osp")) != -1)
    {

        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
        	case 't':
				trainingFile = optarg;
				break;
			case 'T':
				testingFile = optarg;
				Tvalue = testingFile;
				Tflag=true;
				break;
			case 'i':
				iterations = atoi(optarg);
				break;
			case 'l':
				hiddenLayers = atoi(optarg);
				break;
			case 'h':
				neuronsHiddenLayers = atoi(optarg);
				break;
			case 'e':
				eta = atof(optarg);
				break;
			case 'm':
				mu = atof(optarg);
				break;
			case 'v':
				validationRatio = atof(optarg);
				break;
			case 'd':
				decreasingFactor = atof(optarg);
				break;
			case 'o':
				online = true;
				break;
			case 'f':
				functionUsed = atoi(optarg);
				break;
			case 's':
				softmax = 1;
				break;
			case 'w':
				wflag = true;
				wvalue = optarg;
				break;
			case 'p':
				pflag = true;
				break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p')
                    fprintf (stderr, "La opción -%c requiere un argumento.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Opción desconocida `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Caracter de opción desconocido `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }
    if(Tflag == false){
        testingFile = trainingFile;
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value
    	mlp.eta = eta;		//Done!
		mlp.decrementFactor = decreasingFactor;
		mlp.mu = mu;
		mlp.validationRatio = validationRatio;
		mlp.outputFunction = softmax;
		mlp.online = online;

    	// Type of error considered
    	int error=functionUsed; // Done!

    	// Maximum number of iterations
    	int maxIter=iterations; // Done!

        // Read training and test data: call to mlp.readData(...)
    	Dataset * trainDataset = mlp.readData(trainingFile); // Done!
    	Dataset * testDataset = mlp.readData(testingFile); // Done!

    	// Initialize topology vector
        int *topology = new int[hiddenLayers+2];
        topology[0] = trainDataset->nOfInputs;
        for(int i=1; i<(hiddenLayers+2-1); i++)
            topology[i] = neuronsHiddenLayers;
        topology[hiddenLayers+2-1] = trainDataset->nOfOutputs;
        mlp.initialize(hiddenLayers+2,topology);

		// Seed for random numbers
		int seeds[] = {1,2,3,4,5};
		double *trainErrors = new double[5];
		double *testErrors = new double[5];
		double *trainCCRs = new double[5];
		double *testCCRs = new double[5];
		double bestTestError = DBL_MAX;

		double sumTrainError = 0.0, sumTestError = 0.0, sumTrainCCR=0.0, sumTestCCR=0.0;

		for(int i=0; i<5; i++){
			cout << "**********" << endl;
			cout << "SEED " << seeds[i] << endl;
			cout << "**********" << endl;
			srand(seeds[i]);
			mlp.runBackPropagation(trainDataset,testDataset,maxIter,&(trainErrors[i]),&(testErrors[i]),&(trainCCRs[i]),&(testCCRs[i]),error);

			sumTestError +=testErrors[i];
			sumTrainError += trainErrors[i];
			sumTrainCCR += trainCCRs[i];
			sumTestCCR += testCCRs[i];

			cout << "We end!! => Final test CCR: " << testCCRs[i] << endl;

			// We save the weights every time we find a better model
			if(wflag && testErrors[i] <= bestTestError)
			{
				mlp.saveWeights(wvalue);
				bestTestError = testErrors[i];
			}
		}


		double trainAverageError = 0, trainStdError = 0;
		double testAverageError = 0, testStdError = 0;
		double trainAverageCCR = 0, trainStdCCR = 0;
		double testAverageCCR = 0, testStdCCR = 0;

        // Obtain training and test averages and standard deviations
		testAverageError = sumTestError/5;
		trainAverageError = sumTrainError/5;
		trainAverageCCR = sumTrainCCR/5;
		testAverageCCR = sumTestCCR/5;

		for(int i=0; i<5; i++){
			testStdError+=pow((testErrors[i]-testAverageError),2);
			trainStdError+=pow((trainErrors[i]-trainAverageError),2);
			trainStdCCR+=pow((trainCCRs[i]-trainAverageCCR),2);
			testStdCCR+=pow((testCCRs[i]-testAverageCCR),2);

		}
		testStdError = sqrt(testStdError/5);
		trainStdError = sqrt(trainStdError/5);
		trainStdCCR = sqrt(trainStdCCR/5);
		testStdCCR = sqrt(testStdCCR/5);
		cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

		cout << "FINAL REPORT" << endl;
		cout << "*************" << endl;
	    cout << "Train error (Mean +- SD): " << trainAverageError << " +- " << trainStdError << endl;
	    cout << "Test error (Mean +- SD): " << testAverageError << " +- " << testStdError << endl;
	    cout << "Train CCR (Mean +- SD): " << trainAverageCCR << " +- " << trainStdCCR << endl;
	    cout << "Test CCR (Mean +- SD): " << testAverageCCR << " +- " << testStdCCR << endl;
		return EXIT_SUCCESS;
    } else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // You do not have to modify anything from here.
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to mlp.readData(...)
        Dataset *testDataset;
        testDataset = mlp.readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;

	}
}

