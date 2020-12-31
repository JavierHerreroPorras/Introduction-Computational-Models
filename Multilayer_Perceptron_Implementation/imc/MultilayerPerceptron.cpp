/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>
#include <algorithm>		//std::copy()


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	eta = 0.1;
	mu = 0.9;
	validationRatio = 0.0;
	decrementFactor = 1;
	nOfLayers = 3;
	layers = NULL;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {

	nOfLayers = nl;
	layers = new Layer[nOfLayers];

	for(int l = 0; l < nOfLayers; l++){
		layers[l].nOfNeurons = npl[l];
		layers[l].neurons = new Neuron[layers[l].nOfNeurons];
		layers[l].learningRate = eta*pow(decrementFactor,-1*(nOfLayers-l+1));	//Learning rate for every layer

		if(l>0){
			for(int n=0; n < layers[l].nOfNeurons; n++){
				layers[l].neurons[n].w = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].deltaW = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].lastDeltaW = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].wCopy = new double[layers[l-1].nOfNeurons+1];
			}
		}
	}
	cout<<endl;
	return 1;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {

	for(int l = 0; l < nOfLayers; l++){
		if(l>0){
			for(int n=0; n < layers[l].nOfNeurons; n++){
				delete(layers[l].neurons[n].w);
				delete(layers[l].neurons[n].deltaW);
				delete(layers[l].neurons[n].lastDeltaW);
				delete(layers[l].neurons[n].wCopy);
			}
		}
		delete(layers[l].neurons);
	}
	delete(layers);
}

// ------------------------------
// Feel all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {

	for(int l = 1; l < nOfLayers; l++){
		for(int n=0; n < layers[l].nOfNeurons; n++){
			for(int nPL = 0; nPL < layers[l-1].nOfNeurons+1; nPL++){	//nPL = neurons of the Previous Layer
				layers[l].neurons[n].w[nPL] = -1+(rand()/double(RAND_MAX))*2;	//Random values between -1 and 1
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {

	for(int n=0; n < layers[0].nOfNeurons; n++){
		layers[0].neurons[n].out = input[n];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{
	for(int n=0; n < layers[nOfLayers-1].nOfNeurons; n++){
		 output[n] = layers[nOfLayers-1].neurons[n].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {

	for(int l = 1; l < nOfLayers; l++){
		for(int n=0; n < layers[l].nOfNeurons; n++){
			copy(layers[l].neurons[n].w,layers[l].neurons[n].w+layers[l-1].nOfNeurons+1, layers[l].neurons[n].wCopy);
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {

	for(int l = 1; l < nOfLayers; l++){
		for(int n=0; n < layers[l].nOfNeurons; n++){
			copy(layers[l].neurons[n].wCopy,layers[l].neurons[n].wCopy+layers[l-1].nOfNeurons+1, layers[l].neurons[n].w);
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
	
	double net = 0.0;
	for(int l=1; l < nOfLayers; l++){
		for(int n = 0; n < layers[l].nOfNeurons; n++){
			net = 0.0;
			for(int nPL = 0; nPL < layers[l-1].nOfNeurons; nPL++){
				net += layers[l].neurons[n].w[nPL]*layers[l-1].neurons[nPL].out;
			}
			//Bias is set on the last position of the vector
			net+= layers[l].neurons[n].w[layers[l-1].nOfNeurons];
			layers[l].neurons[n].out = sigmoidFunction(net);
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
double MultilayerPerceptron::obtainError(double* target) {

	double sum=0.0;
	for(int n = 0; n < layers[nOfLayers-1].nOfNeurons; n++){
		sum+=pow(target[n] - layers[nOfLayers-1].neurons[n].out,2);
	}

	return double(sum/layers[nOfLayers-1].nOfNeurons);

}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
void MultilayerPerceptron::backpropagateError(double* target) {
	
	for(int n = 0; n < layers[nOfLayers-1].nOfNeurons; n++){
		layers[nOfLayers-1].neurons[n].delta = -1*(target[n]-layers[nOfLayers-1].neurons[n].out)*layers[nOfLayers-1].neurons[n].out*(1-layers[nOfLayers-1].neurons[n].out);
	}

	double sumWeightsDelta = 0.0;
	for(int l=nOfLayers-2; l > 0; l--){
		for(int n = 0; n < layers[l].nOfNeurons; n++){
			sumWeightsDelta = 0.0;
			for(int nNL = 0; nNL < layers[l+1].nOfNeurons; nNL++){
				sumWeightsDelta += layers[l+1].neurons[nNL].w[n]*layers[l+1].neurons[nNL].delta;
			}
			layers[l].neurons[n].delta = sumWeightsDelta * layers[l].neurons[n].out * (1 - layers[l].neurons[n].out);
			}
	}
}


// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {

	for(int l=1; l < nOfLayers; l++){
		for(int n = 0; n < layers[l].nOfNeurons; n++){
			for(int nPL = 0; nPL < layers[l-1].nOfNeurons; nPL++){
				layers[l].neurons[n].deltaW[nPL] =
						layers[l].neurons[n].deltaW[nPL] +
						(layers[l].neurons[n].delta*layers[l-1].neurons[nPL].out);
			}
			layers[l].neurons[n].deltaW[layers[l-1].nOfNeurons] = layers[l].neurons[n].deltaW[layers[l-1].nOfNeurons] + layers[l].neurons[n].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {

	for(int l=1; l < nOfLayers; l++){
		for(int n = 0; n < layers[l].nOfNeurons; n++){
			for(int nPL = 0; nPL < layers[l-1].nOfNeurons; nPL++){
				layers[l].neurons[n].w[nPL] = layers[l].neurons[n].w[nPL] - layers[l].learningRate*layers[l].neurons[n].deltaW[nPL] - mu * layers[l].learningRate * layers[l].neurons[n].lastDeltaW[nPL];
			}
			layers[l].neurons[n].w[layers[l-1].nOfNeurons] = layers[l].neurons[n].w[layers[l-1].nOfNeurons] - layers[l].learningRate*layers[l].neurons[n].deltaW[layers[l-1].nOfNeurons] - mu * layers[l].learningRate * layers[l].neurons[n].lastDeltaW[layers[l-1].nOfNeurons];
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {

	cout<<"====NETWORK===="<<endl;
	for(int l=1; l < nOfLayers; l++){
		cout<<"Layer "<<l<<endl<<"------------"<<endl;
		for(int n = 0; n < layers[l].nOfNeurons; n++){
			cout<<layers[l].neurons[n].w[layers[l-1].nOfNeurons]<<" ";
			for(int nPL = 0; nPL < layers[l-1].nOfNeurons; nPL++){
				cout<<layers[l].neurons[n].w[nPL]<<" ";
			}
			cout<<endl;
		}
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
void MultilayerPerceptron::performEpochOnline(double* input, double* target) {

	//Set deltaW = 0 and save its value in lastDeltaW
	for(int l=1; l < nOfLayers; l++){
		for(int n = 0; n < layers[l].nOfNeurons; n++){
			for(int nPL = 0; nPL < layers[l-1].nOfNeurons + 1; nPL++){
				layers[l].neurons[n].lastDeltaW[nPL] = layers[l].neurons[n].deltaW[nPL];
				layers[l].neurons[n].deltaW[nPL] = 0.0;
			}
		}
	}
	feedInputs(input);
	forwardPropagate();
	backpropagateError(target);
	accumulateChange();
	weightAdjustment();
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset* MultilayerPerceptron::readData(const char *fileName) {

	// Object for reading a file
	ifstream f(fileName);

	if(!f.is_open())
		return NULL;

	Dataset *genericDataset = new Dataset[1];
	int inputs, outputs, patterns;
	// Read number of inputs, outputs and patterns
	f >> inputs >> outputs >> patterns;

	genericDataset = allocateDatasetMemory(inputs, outputs, patterns);

	//Read inputs and outputs of every pattern
	for(int i = 0; i < genericDataset->nOfPatterns; i++){
		for(int j = 0; j < genericDataset->nOfInputs; j++){
			f >> genericDataset->inputs[i][j];
		}

		for(int k = 0; k < genericDataset->nOfOutputs; k++){
			f >> genericDataset->outputs[i][k];
		}
	}

	f.close();

	return genericDataset;
}

// ------------------------------
// Perform an online training for a specific trainDataset
void MultilayerPerceptron::trainOnline(Dataset* trainDataset) {
	int i;
	for(i=0; i<trainDataset->nOfPatterns; i++){
		performEpochOnline(trainDataset->inputs[i], trainDataset->outputs[i]);
	}
}

// ------------------------------
// Test the network with a dataset and return the MSE
double MultilayerPerceptron::test(Dataset* testDataset) {

	double sum = 0.0;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		sum += obtainError(testDataset->outputs[i]);
	}
	return double(sum/testDataset->nOfPatterns);
}


// Optional - KAGGLE
// Test the network with a dataset and return the MSE
// Your have to use the format from Kaggle: two columns (Id y predictied)
void MultilayerPerceptron::predict(Dataset* pDatosTest)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * obtained = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nOfPatterns; i++){

		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(obtained);
		
		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << obtained[j];
		cout << endl;

	}
}

// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
void MultilayerPerceptron::runOnlineBackPropagation(Dataset * trainDataset, Dataset * pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving=0, validationIterWithoutImproving=0;
	double testError = 0, trainError = 0;

	double validationError = 0, lastValidationError=0;
	Dataset *validationDataset = NULL;
	Dataset *auxDataset = NULL;

	// Generate validation data
	if(validationRatio > 0 && validationRatio < 1){

		validationDataset = allocateDatasetMemory(trainDataset->nOfInputs, trainDataset->nOfOutputs, validationRatio * trainDataset->nOfPatterns);

		auxDataset = allocateDatasetMemory(trainDataset->nOfInputs, trainDataset->nOfOutputs, (1-validationRatio) * trainDataset->nOfPatterns);

		//Select index patterns from trainDataset
		int *numbersSelected = new int[validationDataset->nOfPatterns];
		numbersSelected = integerRandomVectoWithoutRepeating(0,trainDataset->nOfPatterns-1, validationDataset->nOfPatterns);
		sort(numbersSelected,numbersSelected+validationDataset->nOfPatterns);

		int n=0,m=0;

		for(int i=0; i < trainDataset->nOfPatterns; i++){
			//Copy selected patterns in validationDataset
			if(n<validationDataset->nOfPatterns && i==numbersSelected[n]){
				for (int y=0; y<trainDataset->nOfInputs; y++){
					validationDataset->inputs[n][y] = trainDataset->inputs[i][y];
				}
				for (int t=0; t<trainDataset->nOfOutputs; t++){
					validationDataset->outputs[n][t] = trainDataset->outputs[i][t];
				}
				//copy(trainDataset->inputs[i],trainDataset->inputs[i]+trainDataset->nOfInputs, validationDataset->inputs[n]);
				//copy(trainDataset->outputs[i],trainDataset->outputs[i]+trainDataset->nOfOutputs, validationDataset->outputs[n]);
				n++;

			}
			//Copy no-selected patterns in auxDataset
			else if(m<auxDataset->nOfPatterns){
				for (int y=0; y<trainDataset->nOfInputs; y++){
					auxDataset->inputs[m][y] = trainDataset->inputs[i][y];
				}
				for (int t=0; t<trainDataset->nOfOutputs; t++){
					auxDataset->outputs[m][t] = trainDataset->outputs[i][t];
				}
				m++;
			}

		}
	}
	else{
		auxDataset = trainDataset;
	}

	// Learning
	do {
		//Early stoping with validation version
		if(validationRatio > 0 && validationRatio < 1){
			trainOnline(auxDataset);
			lastValidationError = validationError;
			validationError = test(validationDataset);
			trainError = test(auxDataset);

			if(countTrain==0 || validationError < lastValidationError){
				copyWeights();
				validationIterWithoutImproving = 0;
			}
			else if( (validationError-lastValidationError) < 0.00001)
				validationIterWithoutImproving = 0;
			else
				validationIterWithoutImproving++;

			if(countTrain==0 || trainError < minTrainError){
				minTrainError = trainError;
				copyWeights();
				iterWithoutImproving = 0;
			}
			else if( (trainError-minTrainError) < 0.00001)
				iterWithoutImproving = 0;
			else
				iterWithoutImproving++;

			if(validationIterWithoutImproving==50 || iterWithoutImproving==50){
				cout << "We exit because the training is not improving!!"<< endl;
				restoreWeights();
				countTrain = maxiter;
			}
		}

		//Early stoping with standard version
		else{
			trainOnline(auxDataset);
			trainError = test(auxDataset);
			if(countTrain==0 || trainError < minTrainError){
				minTrainError = trainError;
				copyWeights();
				iterWithoutImproving = 0;
			}
			else if( (trainError-minTrainError) < 0.00001)
				iterWithoutImproving = 0;
			else
				iterWithoutImproving++;

			if(iterWithoutImproving==50){
				cout << "We exit because the training is not improving!!"<< endl;
				restoreWeights();
				countTrain = maxiter;
			}
		}

		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while ( countTrain<maxiter );


	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nOfPatterns; i++){
		double* prediction = new double[pDatosTest->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(pDatosTest->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<pDatosTest->nOfOutputs; j++)
			cout << pDatosTest->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	testError = test(pDatosTest);
	*errorTest=testError;
	*errorTrain=minTrainError;

	//freeDatasetMemory(validationDataset);
	//freeDatasetMemory(auxDataset);

}

// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * archivo)
{
	// Object for writing the file
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
		f << " " << layers[i].nOfNeurons;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * archivo)
{
	// Object for reading a file
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
		f >> npl[i];

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
