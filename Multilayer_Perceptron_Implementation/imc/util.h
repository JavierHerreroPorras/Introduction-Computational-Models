/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <math.h>	//	exp()
#include <algorithm>	//	swap()
#include "MultilayerPerceptron.h"

namespace util{
static int * integerRandomVectoWithoutRepeating(int min, int max, int howMany){
        int total = max-min+1;
        int* numbersToBeSelected = new int[total];
        int* numbersSelected = new int[howMany];
        // Initialize the list of possible selections
        for(int i = 0; i < total; i++)
                numbersToBeSelected[i] = min+i;

        for(int i=0; i < howMany; i++)
        {
                int selectedNumber = rand() % (total-i);
                // Store the selected number
                numbersSelected[i]=numbersToBeSelected[selectedNumber];
                // We include the last valid number in numbersToBeSelected, in this way
                // all numbers are valid until total-i-1
                numbersToBeSelected[selectedNumber]=numbersToBeSelected[total-i-1];
        }
        delete [] numbersToBeSelected;
        return numbersSelected;

};

}

double sigmoidFunction(double net){
	return double(1/(1+exp(-net)));
}



imc::Dataset *allocateDatasetMemory(int nOfInputs, int nOfOutputs, int nOfPatterns){
	imc::Dataset *genericDataset = new imc::Dataset[1];
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

void freeDatasetMemory(imc::Dataset *dataset){
	for(int p = 1; p < dataset->nOfPatterns; p++){
		delete(dataset->inputs[p]);
		delete(dataset->outputs[p]);
	}
	delete(dataset->inputs);
	delete(dataset->outputs);
	delete(dataset);
}





#endif /* UTIL_H_ */
