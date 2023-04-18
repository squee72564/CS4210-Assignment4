#-------------------------------------------------------------------------
# AUTHOR: Alexander Rodriguez 
# FILENAME: perceptron.py 
# SPECIFICATION: Perceptron classifiers
# FOR: CS 4210- Assignment #4
# TIME SPENT: 10-15 mins
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
from sklearn.exceptions import ConvergenceWarning
import warnings
import numpy as np
import pandas as pd
# I suppressed convergence warnings to make the output clearer
warnings.filterwarnings("ignore", category=ConvergenceWarning)

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

highest_perceptron_accuracy = 0
highest_mlp_accuracy = 0

for rate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms
        for algorithm in ['perceptron', 'mlp']:

            #Create a Neural Network classifier
            if algorithm == 'perceptron':
                clf = Perceptron(eta0=rate, shuffle=shuffle, max_iter=1000)    
            else:
                clf = MLPClassifier(activation='logistic', learning_rate_init=rate, hidden_layer_sizes=(10,), shuffle=shuffle, max_iter=1000)
            
            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            accuracy = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                if prediction == y_testSample:
                    accuracy += 1
            accuracy /= len(y_test)

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. 
            #If so, update the highest accuracy and print it together with the network hyperparameters
            if algorithm == 'perceptron' and accuracy > highest_perceptron_accuracy:
                highest_perceptron_accuracy = accuracy
                print(f"Highest Perceptron accuracy so far: {accuracy:.2f}, Parameters: learning rate={rate:.4f}, shuffle={shuffle}")
            elif algorithm == 'mlp' and accuracy > highest_mlp_accuracy:
                highest_mlp_accuracy = accuracy
                print(f"Highest MLP accuracy so far: {accuracy:.2f}, Parameters: learning rate={rate:.4f}, shuffle={shuffle}")
