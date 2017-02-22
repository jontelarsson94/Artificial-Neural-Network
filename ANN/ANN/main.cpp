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
using namespace std;

int numOfNeurons = 1500;
float input[2200][3];
float hiddenWeights[3];
float hiddenOutputs[3];
float weight[3][3];
float expectedOutput[2200];

const float learningRate = 0.2;

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
    weight[0][0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weight[0][1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weight[0][2] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weight[1][0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weight[1][1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weight[1][2] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weight[2][0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weight[2][1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    weight[2][2] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    hiddenWeights[0] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    hiddenWeights[1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    hiddenWeights[2] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

float calculateNet(int pos, int weightPos)
{
    float net = 0.0;
    
    net += input[pos][0] * weight[weightPos][0];
    net += input[pos][1] * weight[weightPos][1];
    net += input[pos][2] * weight[weightPos][2];
    
    return net;
}

float calculateHiddenNet(int pos)
{
    float net = 0.0;
    
    net += hiddenOutputs[0] * hiddenWeights[0];
    net += hiddenOutputs[1] * hiddenWeights[1];
    net += hiddenOutputs[2] * hiddenWeights[2];
    
    return net;
}

float sigmoid(float net)
{
    return 1 / (1 + exp(-net));
}

float calculateOutputError(float output, int pos)
{
    return (1 - output) * output * (expectedOutput[pos] - output);
}

void setInputWeights(int pos, int weightPos, float error)
{
    weight[weightPos][0] = learningRate*error*input[pos][0];
    weight[weightPos][1] = learningRate*error*input[pos][1];
    weight[weightPos][2] = learningRate*error*input[pos][2];
}

void setHiddenValue(float output, int j)
{
    hiddenOutputs[j] = output;
}

void setHiddenWeights(float error)
{
    hiddenWeights[0] = learningRate * error * hiddenOutputs[0];
    hiddenWeights[1] = learningRate * error * hiddenOutputs[1];
    hiddenWeights[2] = learningRate * error * hiddenOutputs[2];
}

float calculateHiddenError(int i, float outputError)
{
    return hiddenOutputs[i] * (1- hiddenOutputs[i]) * hiddenWeights[i] * outputError;
}

int main(int argc, const char * argv[]) {
    
    srand(NULL);
    readData("titanic.txt");
    initializeWeights();
    
    float error = 0.0;
    float net, output, lastError = 1.0;
    int count;
    int correct = 0;
    while(correct < 500) {
        for(int i = 0; i < 1500; i++) {
            count = 0;
            error = numeric_limits<float>::max();
            lastError = error;
            error = 0.0;
            for(int j = 0; j < 3; j++) {
                net = calculateNet(i, j);
                output = sigmoid(net);
                setHiddenValue(output, j);
            }
            net = calculateHiddenNet(i);
            output = sigmoid(net);
            
            float outputError = calculateOutputError(output, i);
            error += outputError;
            
            setHiddenWeights(outputError);
            
            for(int j = 0; j < 3; j++) {
                float hiddenError = calculateHiddenError(j, outputError);
                error += hiddenError;
                
                setInputWeights(i, j, hiddenError);
            }
            error = error/4;
            //cout << error << endl;
        }
        
        
        correct = 0;
        int uncorrect = 0;
        int over = 0;
        
        for(int i = 1500; i < 2200; i++) {
            for(int j = 0; j < 3; j++) {
                net = calculateNet(i, j);
                output = sigmoid(net);
                setHiddenValue(output, j);
            }
            net = calculateHiddenNet(i);
            output = sigmoid(net);
            if(output == 0.5)
                over++;
            //cout << output << endl;
            
            if(expectedOutput[i] == 1.0){
                if(output >= 0.5)
                    correct++;
                else
                    uncorrect++;
            } else {
                if(output < 0.5)
                    correct++;
                else
                    uncorrect++;
            }
        }
        
        cout << "Correct: " << correct << endl;
        cout << "Uncorrect: " << uncorrect << endl;
    }
    
    
    
    return 0;
}
