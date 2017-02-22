//
//  main.cpp
//  ANN
//
//  Created by Jonathan Larsson on 2017-02-22.
//  Copyright © 2017 Jonathan Larsson. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <fstream>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <cmath>
#include <unistd.h>
using namespace std;

int numOfNeurons = 1500;
float input[2200][3];
float weightsToHidden[3][3];
float hiddenOutputs[3];
float weightsToOutput[3];
float expectedOutput[2200];

const float learningRate = 0.1;

void readData(string fileName)
{
    ifstream infile(fileName);
    float x1, x2, x3, output;
    char afterFirst, afterSecond, afterThird;
    int i = 0;
    int j = 0;
    while (infile >> x1 >> afterFirst >> x2 >> afterSecond >> x3 >> afterThird >> output)
    {
        if (i < numOfNeurons) {
            input[i][0] = x1;
            input[i][1] = x2;
            input[i][2] = x3;
            if(output < 0)
                expectedOutput[i] = 0.25;
            else
                expectedOutput[i] = 0.75;
        } else {
            input[i][0] = x1;
            input[i][1] = x2;
            input[i][2] = x3;
            expectedOutput[i] = output;
        }
        i++;
    }
    
}

void initializeWeights()
{
    weightsToHidden[0][0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToHidden[0][1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToHidden[0][2] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToHidden[1][0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToHidden[1][1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToHidden[1][2] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToHidden[2][0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToHidden[2][1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToHidden[2][2] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToOutput[0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToOutput[1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weightsToOutput[2] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

float sigmoid(float net)
{
    return 1 / (1 + exp(-net));
}

float calculateHiddenNet(int i, int j)
{
    return (input[i][0] * weightsToHidden[0][j]) + (input[i][1] * weightsToHidden[1][j]) + (input[i][2] * weightsToHidden[2][j]);
}

float calculateOutputNet()
{
    return (hiddenOutputs[0] * weightsToOutput[0]) + (hiddenOutputs[1] * weightsToOutput[1]) + (hiddenOutputs[2] * weightsToOutput[2]);
}

float calculateOutputError(int i, float output)
{
    return output * (1 - output) * (expectedOutput[i] - output);
}

float calculateWeightToOutput(int j, float outputError)
{
    return learningRate * outputError * hiddenOutputs[j];
}

float calculateHiddenError(int j, float outputError)
{
    return hiddenOutputs[j] * (1 - hiddenOutputs[j]) * weightsToOutput[j] * outputError;
}

float calculateWeightToHidden(int i, int k, float hiddenError)
{
    return learningRate * hiddenError * input[i][k];
}

int main(int argc, const char * argv[]) {
    
    srand(NULL);
    readData("titanic.txt");
    initializeWeights();
    float net, output, outputError;
    float hiddenError[3];
    float worstOutput = 5.0;
    while(abs(worstOutput) > 0.000001){
        cout << worstOutput << endl;
        worstOutput = 0.0;
        for(int i = 0; i < 1500; i++) {
            outputError = 5.0;
            //Calculate output at hidden layer
            for(int j = 0; j < 3; j++) {
                net = calculateHiddenNet(i, j);
                hiddenOutputs[j] = sigmoid(net);
            }
                    
            //Calculate output at output layer
            net = calculateOutputNet();
            output = sigmoid(net);
                    
            //calculate error at output layer
            outputError = calculateOutputError(i, output);
            
            if(outputError > worstOutput)
                worstOutput = outputError;
            
            
            //calculate error at hidden layer
            for(int j = 0; j < 3; j++) {
                hiddenError[j] = calculateHiddenError(j, outputError);
            }
                    
            //calculate new weights for paths from hidden to output
            //Better if accumulating then just setting
            for(int j = 0; j < 3; j++) {
                weightsToOutput[j] += calculateWeightToOutput(j, outputError);
            }
                    
            //calculate new weights for paths from input to hidden
            //Better if accumulating than just setting
            for(int j = 0; j < 3; j++) {
                for(int k = 0; k < 3; k++) {
                    weightsToHidden[k][j] += calculateWeightToHidden(i, k, hiddenError[j]);
                }
            }
        }
    }
    
    int correct = 0;
    int uncorrect = 0;
    
    for(int i = 1500; i < 2200; i++) {
        for(int j = 0; j < 3; j++) {
            net = calculateHiddenNet(i, j);
            hiddenOutputs[j] = sigmoid(net);
        }
        
        //Calculate output at output layer
        net = calculateOutputNet();
        output = sigmoid(net);
        
        if(expectedOutput[i] == 1.0 && output >= 0.5)
            correct++;
        else if(expectedOutput[i] == 1.0 && output < 0.5)
            uncorrect++;
        if(expectedOutput[i] == -1.0 && output >= 0.5)
            uncorrect++;
        else if(expectedOutput[i] == -1.0 && output < 0.5)
            correct++;
        
    }

    cout << correct << endl;
    cout << uncorrect << endl;
    
    
    
    return 0;
}
