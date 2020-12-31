//============================================================================
// Introduction to computational models
// Name        : la1.cpp
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

#include "imc/MultilayerPerceptron.h"


using namespace imc;
using namespace std;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool wflag = 0, pflag = 0, Tflag= 0;
    char *Tvalue = NULL, *wvalue = NULL;
    char *trainingFile = NULL, *testingFile = NULL;
    int c, iterations = 1000, hiddenLayers = 1, neuronsHiddenLayers = 5;
    double eta = 0.1, mu = 0.9, validationRatio = 0.0, decreasingFactor = 1;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:v:d:w:p")) != -1)
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
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt == 't')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
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

        // Parameters of the mlp. For example, mlp.eta = value;
    	mlp.eta = eta;		//Done!
    	mlp.decrementFactor = decreasingFactor;
    	mlp.mu = mu;
    	mlp.validationRatio = validationRatio;

        // Read training and test data: call to mlp.readData(...)
    	Dataset * trainDataset = mlp.readData(trainingFile); // Done!
    	Dataset * testDataset = mlp.readData(testingFile); // Done!

        // Initialize topology vector
    	int layers=hiddenLayers + 2; // Done!
    	int * topology= new int[layers]; // Done!
    	topology[0] = trainDataset->nOfInputs;
    	for(int i = 1; i < layers-1; i++){
    		topology[i] = neuronsHiddenLayers;
    	}
    	topology[layers-1] = trainDataset->nOfOutputs;

        // Initialize the network using the topology vector
        mlp.initialize(layers,topology);


        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = 1;

        double sumTrainError = 0.0, sumTestError = 0.0;

        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset,testDataset,iterations,&(trainErrors[i]),&(testErrors[i]));

            sumTestError +=testErrors[i];
            sumTrainError += trainErrors[i];
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;

        averageTestError = sumTestError/5;
        averageTrainError = sumTrainError/5;

        for(int i=0; i<5; i++){
        	stdTestError+=pow((testErrors[i]-averageTestError),2);
        	stdTrainError+=pow((trainErrors[i]-averageTrainError),2);
        }
        stdTestError = sqrt(stdTestError/5);
        stdTrainError = sqrt(stdTrainError/5);

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
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

