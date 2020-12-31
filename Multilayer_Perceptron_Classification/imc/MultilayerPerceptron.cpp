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
#include <algorithm>


using namespace imc;
using namespace std;
using namespace util;


// ------------------------------
// Allocate memory for a Dataset
Dataset *allocateDatasetMemory(int nOfInputs, int nOfOutputs, int nOfPatterns){
	Dataset *genericDataset = new Dataset;
	genericDataset->nOfInputs = nOfInputs;
	genericDataset->nOfOutputs = nOfOutputs;
	genericDataset->nOfPatterns = nOfPatterns;

	genericDataset->inputs = new double*[genericDataset->nOfPatterns];
	genericDataset->outputs = new double*[genericDataset->nOfPatterns];

	for(int i = 0; i < genericDataset->nOfPatterns; i++){
		genericDataset->inputs[i] = new double[genericDataset->nOfInputs];
		genericDataset->outputs[i] = new double[genericDataset->nOfOutputs];
	}

	return genericDataset;

}

// ------------------------------
// Check if i == j
int I(int i, int j){
	return i==j;
}

// ------------------------------
// Return the index of the max value in the vector
int argmax(double v[], int n){
	int maxindex=0;
	for(int i=0; i < n; i++){
		if(v[maxindex] < v[i]){
			maxindex = i;
		}
	}
	return maxindex;
}

// ------------------------------
// Obtain an integer random number in the range [Low,High]
int randomInt(int Low, int High)
{
	return 1;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double randomDouble(double Low, double High)
{
	return 1;
}

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	this->nOfLayers = 3;
	this->decrementFactor = 1;
	this->eta = 0.1;
	this->mu = 0.9;
	this->validationRatio = 0.0;
	this->nOfTrainingPatterns = 0;
	this->online = false;
	this->outputFunction = 0;
	this->confusionMatrix = NULL;
	this->confusionMatrixDimensions=0;
	layers = NULL;
}

// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {

	int neurons;
	nOfLayers = nl;
	layers = new Layer[nOfLayers];

	for(int l = 0; l < nOfLayers; l++){
		layers[l].nOfNeurons = npl[l];
		layers[l].neurons = new Neuron[layers[l].nOfNeurons];

		if(l>0 && l < nOfLayers-1){
			for(int n=0; n < layers[l].nOfNeurons; n++){
				layers[l].neurons[n].w = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].deltaW = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].lastDeltaW = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].wCopy = new double[layers[l-1].nOfNeurons+1];
				for(int nPL = 1; nPL < layers[l-1].nOfNeurons+1; nPL++){
					layers[l].neurons[n].deltaW[nPL] = 0;
					layers[l].neurons[n].w[nPL] = 0;
					layers[l].neurons[n].lastDeltaW[nPL] = 0;
					layers[l].neurons[n].wCopy[nPL] = 0;
				}
			}
		}

		else if (l == nOfLayers - 1){
			neurons = layers[l].nOfNeurons;
			//If we are using Softmax, last neuron of the output layer is not used, its memory is not allocated
			if(outputFunction == 1){
				neurons -= 1;
				layers[l].neurons[layers[l].nOfNeurons-1].w = NULL;
				layers[l].neurons[layers[l].nOfNeurons-1].deltaW = NULL;
				layers[l].neurons[layers[l].nOfNeurons-1].lastDeltaW = NULL;
				layers[l].neurons[layers[l].nOfNeurons-1].wCopy = NULL;
				layers[l].neurons[layers[l].nOfNeurons-1].out = 0.0;
			}

			for(int n=0; n < neurons; n++){
				layers[l].neurons[n].w = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].deltaW = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].lastDeltaW = new double[layers[l-1].nOfNeurons+1];
				layers[l].neurons[n].wCopy = new double[layers[l-1].nOfNeurons+1];

				for(int nPL = 1; nPL < layers[l-1].nOfNeurons+1; nPL++){
					layers[l].neurons[n].deltaW[nPL] = 0;
					layers[l].neurons[n].w[nPL] = 0;
					layers[l].neurons[n].lastDeltaW[nPL] = 0;
					layers[l].neurons[n].wCopy[nPL] = 0;
				}
			}


		}
	}
	//Allocate memory for confusion matrix
	confusionMatrixDimensions = layers[nOfLayers-1].nOfNeurons;
	confusionMatrix = new int*[confusionMatrixDimensions];

	for (int i=0; i < confusionMatrixDimensions; i++){
		confusionMatrix[i] = new int[confusionMatrixDimensions];
	}

	//Put all confusion matrix elements to 0
	for (int i=0; i < confusionMatrixDimensions; i++){
		for (int j=0; j < confusionMatrixDimensions; j++){
			confusionMatrix[i][j]=0;
		}
	}

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
			int neurons = layers[l].nOfNeurons;

			if(l == nOfLayers-1 && outputFunction == 1){
				neurons = layers[l].nOfNeurons-1;
			}

			for(int n=0; n < neurons; n++){
				delete(layers[l].neurons[n].w);
				delete(layers[l].neurons[n].deltaW);
				delete(layers[l].neurons[n].lastDeltaW);
				delete(layers[l].neurons[n].wCopy);
			}
		}
		delete(layers[l].neurons);
	}
	delete(layers);

	for(int i = 0; i < confusionMatrixDimensions; i++){
		delete(confusionMatrix[i]);
	}
		delete(confusionMatrix);
}

//-------------------------------
// Free memory for a Dataset
void freeDatasetMemory(Dataset* dataset) {

	int nPatterns = dataset->nOfPatterns;

	for(int i = 0; i<nPatterns; i++){
		delete(dataset->inputs[i]);
		delete(dataset->outputs[i]);
	}
	delete(dataset->inputs);
	delete(dataset->outputs);
	delete(dataset);
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {

	for(int l = 1; l < nOfLayers; l++){

		int neurons = layers[l].nOfNeurons;

		if(l == nOfLayers-1 && outputFunction == 1){
			neurons = layers[l].nOfNeurons-1;
		}

		for(int n=0; n < neurons; n++){
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

		int neurons = layers[l].nOfNeurons;

		if(l == nOfLayers-1 && outputFunction == 1){
			neurons = layers[l].nOfNeurons-1;
		}

		for(int n=0; n < neurons; n++){
			copy(layers[l].neurons[n].w,layers[l].neurons[n].w+layers[l-1].nOfNeurons+1, layers[l].neurons[n].wCopy);
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {

	for(int l = 1; l < nOfLayers; l++){
		int neurons = layers[l].nOfNeurons;

		if(l == nOfLayers-1 && outputFunction == 1){
			neurons = layers[l].nOfNeurons-1;
		}

		for(int n=0; n < neurons; n++){
			copy(layers[l].neurons[n].wCopy,layers[l].neurons[n].wCopy+layers[l-1].nOfNeurons+1, layers[l].neurons[n].w);
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {

	double *net;
	double accumulateNet = 0;
	for(int l=1; l < nOfLayers; l++){
		net = new double[layers[l].nOfNeurons];
		accumulateNet=0;
		for(int n = 0; n < layers[l].nOfNeurons; n++){
			net[n] = 0.0;
			//To avoid last neuron of last layer in softmax function --> net will be 0
			if(layers[l].neurons[n].w != NULL){
				for(int nPL = 1; nPL < layers[l-1].nOfNeurons+1; nPL++){
					net[n] += layers[l].neurons[n].w[nPL]*layers[l-1].neurons[nPL-1].out;
				}
				//Bias is set on the first position of the vector
				net[n]+= layers[l].neurons[n].w[0];

				//Using sigmoid activation.
				layers[l].neurons[n].out = double(1/(1+exp(-net[n])));
			}
			//accumulateNet is the sumatory of all neurons net of the layer
			accumulateNet+=exp(net[n]);
		}
		//If using softmax function, accumulateNet has been calculated before calculating neurons output.
		if(l == nOfLayers - 1 && outputFunction == 1){
			for(int n = 0; n < layers[l].nOfNeurons; n++){
				layers[l].neurons[n].out = exp(net[n])/accumulateNet;
			}
		}
		delete net;

	}
}


// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double* target, int errorFunction) {
	double sum=0.0;
	int neurons = layers[nOfLayers-1].nOfNeurons;

	for(int n=0; n < neurons; n++){
		if(errorFunction == 0){
			sum+=pow(target[n] - layers[nOfLayers-1].neurons[n].out,2);
		}
		else{
			sum+=target[n]*log(layers[nOfLayers-1].neurons[n].out);
		}
	}

	return double(sum/layers[nOfLayers-1].nOfNeurons);

}

// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction) {

	int neurons = layers[nOfLayers-1].nOfNeurons;

	if(outputFunction == 1){
		neurons = layers[nOfLayers-1].nOfNeurons-1;
	}

	for(int n=0; n < neurons; n++){
		if(outputFunction == 0){
			if(errorFunction == 0){
				//MSE + SIGMOID
				layers[nOfLayers-1].neurons[n].delta = -1*(target[n]-layers[nOfLayers-1].neurons[n].out)*layers[nOfLayers-1].neurons[n].out*(1-layers[nOfLayers-1].neurons[n].out);
			}else{
				//CROSS ENTROPY + SIGMOID
				layers[nOfLayers-1].neurons[n].delta = -1*(target[n]/layers[nOfLayers-1].neurons[n].out)*layers[nOfLayers-1].neurons[n].out*(1-layers[nOfLayers-1].neurons[n].out);
			}
		}else{
			if(errorFunction == 0){
				//MSE + SOFTMAX
				double sumDelta = 0.0;
				for(int nSL = 0; nSL < layers[nOfLayers-1].nOfNeurons; nSL++){
					sumDelta += (target[nSL]-layers[nOfLayers-1].neurons[nSL].out)*layers[nOfLayers-1].neurons[n].out*(I(nSL,n)-layers[nOfLayers-1].neurons[nSL].out);
				}
				layers[nOfLayers-1].neurons[n].delta = -1*sumDelta;
			}else{
				//CROSS ENTROPY + SOFTMAX
				double sumDelta = 0.0;
				for(int nSL = 0; nSL < layers[nOfLayers-1].nOfNeurons; nSL++){
					sumDelta += (target[nSL]/layers[nOfLayers-1].neurons[nSL].out)*layers[nOfLayers-1].neurons[n].out*(I(nSL,n)-layers[nOfLayers-1].neurons[nSL].out);
				}
				layers[nOfLayers-1].neurons[n].delta = -1*sumDelta;
			}
		}
	}

	//The process is the same than the previous lab assignment
	double sumWeightsDelta = 0.0;
	for(int l=nOfLayers-2; l > 0; l--){
		for(int n=0; n < layers[l].nOfNeurons; n++){
			sumWeightsDelta = 0.0;
			int neurons = layers[l+1].nOfNeurons;

			if(l+1 == nOfLayers-1 && outputFunction == 1){
				neurons = layers[l+1].nOfNeurons-1;
			}

			for(int nNL = 0; nNL < neurons; nNL++){
				sumWeightsDelta += layers[l+1].neurons[nNL].w[n+1]*layers[l+1].neurons[nNL].delta;
			}
			layers[l].neurons[n].delta = sumWeightsDelta * layers[l].neurons[n].out * (1 - layers[l].neurons[n].out);
			}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {

	for(int l=1; l < nOfLayers; l++){
		int neurons = layers[l].nOfNeurons;

		if(l == nOfLayers-1 && outputFunction == 1){
			neurons = layers[l].nOfNeurons-1;
		}

		for(int n=0; n < neurons; n++){
			for(int nPL = 1; nPL < layers[l-1].nOfNeurons+1; nPL++){
				layers[l].neurons[n].deltaW[nPL] += layers[l].neurons[n].delta*layers[l-1].neurons[nPL-1].out;
			}
			layers[l].neurons[n].deltaW[0] += layers[l].neurons[n].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {

	for(int l=1; l < nOfLayers; l++){
		double learningRate = eta*pow(decrementFactor,-1*(nOfLayers-l));	//Learning rate for every layer
		int neurons = layers[l].nOfNeurons;

		if(l == nOfLayers-1 && outputFunction == 1){
			neurons = layers[l].nOfNeurons-1;
		}

		for(int n=0; n < neurons; n++){
			for(int nPL = 1; nPL < layers[l-1].nOfNeurons+1; nPL++){
				if(online){
					layers[l].neurons[n].w[nPL] = layers[l].neurons[n].w[nPL] - learningRate*layers[l].neurons[n].deltaW[nPL] - mu * learningRate * layers[l].neurons[n].lastDeltaW[nPL];
				}else{
					//Offline learning, it is divided by the number of training patterns, to make delta lower.
					layers[l].neurons[n].w[nPL] = layers[l].neurons[n].w[nPL] - (learningRate*layers[l].neurons[n].deltaW[nPL])/nOfTrainingPatterns - (mu * learningRate * layers[l].neurons[n].lastDeltaW[nPL])/nOfTrainingPatterns;
				}
			}
			if(online){
				layers[l].neurons[n].w[0] = layers[l].neurons[n].w[0] - learningRate*layers[l].neurons[n].deltaW[0] - mu * learningRate * layers[l].neurons[n].lastDeltaW[0];
			}else{
				//Offline learning, it is divided by the number of training patterns, to make delta lower.
				layers[l].neurons[n].w[0] = layers[l].neurons[n].w[0] - (learningRate*layers[l].neurons[n].deltaW[0])/nOfTrainingPatterns - (mu * learningRate * layers[l].neurons[n].lastDeltaW[0])/nOfTrainingPatterns;
			}

		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {

	for(int l = 1; l < nOfLayers; l++){

		int neurons = layers[l].nOfNeurons;

		if(l == nOfLayers-1 && outputFunction == 1){
			neurons = layers[l].nOfNeurons-1;
		}
		for(int n = 0; n < neurons; n++){
			for(int nPL = 0; nPL < layers[l-1].nOfNeurons+1; nPL++){
				cout<<layers[l].neurons[n].w[nPL]<<" ";
			}
			cout<<endl<<endl;
		}
	}
	cout<<endl;
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction) {

	if(online){

		//Set deltaW = 0 and save its value in lastDeltaW
		for(int l=1; l < nOfLayers; l++){
			int neurons = layers[l].nOfNeurons;

			if(l == nOfLayers-1 && outputFunction == 1){
				neurons = layers[l].nOfNeurons-1;
			}

			for(int n=0; n < neurons; n++){
				for(int nPL = 0; nPL < layers[l-1].nOfNeurons + 1; nPL++){
					layers[l].neurons[n].lastDeltaW[nPL] = layers[l].neurons[n].deltaW[nPL];
					layers[l].neurons[n].deltaW[nPL] = 0.0;
				}
			}
		}
	}

	feedInputs(input);

	forwardPropagate();

	backpropagateError(target, errorFunction);

	accumulateChange();


	if(online){weightAdjustment();}

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
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {

int i;
	if(!online){

		//Set deltaW = 0 and save its value in lastDeltaW
		for(int l=1; l < nOfLayers; l++){
			int neurons = layers[l].nOfNeurons;

			if(l == nOfLayers-1 && outputFunction == 1){
				neurons = layers[l].nOfNeurons-1;
			}

			for(int n=0; n < neurons; n++){
				for(int nPL = 0; nPL < layers[l-1].nOfNeurons + 1; nPL++){
					layers[l].neurons[n].lastDeltaW[nPL] = layers[l].neurons[n].deltaW[nPL];
					layers[l].neurons[n].deltaW[nPL] = 0.0;
				}
			}
		}
	}
	for(i=0; i<trainDataset->nOfPatterns; i++){
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i],errorFunction);
	}
	if(!online){weightAdjustment();}
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {
	double sum = 0.0;
	for(int i=0; i<dataset->nOfPatterns; i++){
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		sum += obtainError(dataset->outputs[i],errorFunction);
	}

	int aux = 1;
	if(errorFunction == 1){aux = -1;}

	return double(aux*sum/dataset->nOfPatterns);
}

// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {

	double sum = 0;
	double output[dataset->nOfOutputs];
	for(int i=0; i<dataset->nOfPatterns; i++){
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(output);
		int predictedClass = argmax(output, dataset->nOfOutputs);
		int desiredClass = argmax(dataset->outputs[i],dataset->nOfOutputs);
		/*if(predictedClass != desiredClass)
			cout<<"Pattern "<<i<<" was clasified in "<<predictedClass<<" when was "<<desiredClass<<" class."<<endl;
		 */
		sum += predictedClass==desiredClass;
		confusionMatrix[desiredClass][predictedClass]++;
	}

	for (int i=0; i < confusionMatrixDimensions; i++){
		for (int j=0; j < confusionMatrixDimensions; j++){
			//cout<<confusionMatrix[i][j]<<" ";
			confusionMatrix[i][j]=0;
		}
		//cout<<endl;
	}
	return 100*sum/dataset->nOfPatterns;
}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset* dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Category" << endl;
	
	for (i=0; i<dataset->nOfPatterns; i++){

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}



// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();


	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;

	double validationError = 0, previousValidationError = 0;
	int iterWithoutImprovingValidation = 0;
	Dataset *validationDataset = NULL;
	Dataset *auxDataset = NULL;

	/*std::ofstream ofs;
	ofs.open ("error.txt", std::ofstream::out | std::ofstream::app);

	ofs << "\n\n\nNEW SEED\n\n\n";*/

	// Generate validation data
	if(validationRatio > 0 && validationRatio < 1){
		validationDataset = allocateDatasetMemory(trainDataset->nOfInputs, trainDataset->nOfOutputs, validationRatio * trainDataset->nOfPatterns);

		auxDataset = allocateDatasetMemory(trainDataset->nOfInputs, trainDataset->nOfOutputs, (1-validationRatio) * trainDataset->nOfPatterns);

		//Select index patterns from trainDataset
		int *numbersSelected = new int[validationDataset->nOfPatterns];
		numbersSelected = integerRandomVectorWithoutRepeating(0,trainDataset->nOfPatterns-1, validationDataset->nOfPatterns);
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
		train(auxDataset,errorFunction);

		double trainError = test(auxDataset,errorFunction);
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

		countTrain++;

		if(validationDataset!=NULL){
			if(previousValidationError==0)
				previousValidationError = 999999999.9999999999;
			else
				previousValidationError = validationError;
			validationError = test(validationDataset,errorFunction);
			if(validationError < previousValidationError)
				iterWithoutImprovingValidation = 0;
			else if((validationError-previousValidationError) < 0.00001)
				iterWithoutImprovingValidation = 0;
			else
				iterWithoutImprovingValidation++;
			if(iterWithoutImprovingValidation==50){
				cout << "We exit because validation is not improving!!"<< endl;
				restoreWeights();
				countTrain = maxiter;
			}
		}
		/*double Test = testClassification(testDataset);
		double Train = testClassification(auxDataset);
		double Validation = 0;
		if(validationRatio > 0 && validationRatio < 1)
			Validation = testClassification(validationDataset);

		ofs << Train <<" "<< Test <<" "<< Validation << endl;*/

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << "\t Validation error: " << validationError << endl;

	} while ( countTrain<maxiter );


	if ( (iterWithoutImprovingValidation!=50) && (iterWithoutImproving!=50))
		restoreWeights();

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}
	*errorTest=test(testDataset,errorFunction);
	*errorTrain=minTrainError;
	//cout<<"\nTEST CONFUSION MATRIX:"<<endl;
	*ccrTest = testClassification(testDataset);
	//cout<<"\nTRAIN CONFUSION MATRIX:"<<endl;
	*ccrTrain = testClassification(auxDataset);


}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(layers[i].neurons[j].w!=NULL)
				    f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * fileName)
{
	// Object for reading a file
	ifstream f(fileName);

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
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(!(outputFunction==1 && (i==(nOfLayers-1)) && (k==(layers[i].nOfNeurons-1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
