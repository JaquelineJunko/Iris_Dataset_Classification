# Iris_Dataset_Classification
Perceptron for Iris dataset classification. This work was develop using Matlab R2018a 

# Iris Data Set

The dataset used for the experiment is provided by the UCI. There are a total of 150 samples of Iris flowers, divided into three species - setosa, virginica and versicolor. Each of the samples has five attributes, one of which is the class indicator. Thus, each line of the file contains sepal length, sepal width, petal length, petal width and species name, in the given order. There are a total of 50 samples for each of the three classes.

The dataset is available at:

http://archive.ics.uci.edu/ml/datasets/Iris

# General informations about the algorithm

The constructed network consists of a simple single-layer Perceptron with four input neurons - one for each attribute, and three possible outputs - one for each class.

**Network characteristics:** Simple single-layer Perceptron. It is composed of 4 inputs (neurons), one for each attribute (sepal length and width, and petal length and width) It has 3 possible outputs, one for each class.

**Input:**
```
o:          integer     -> Number of outputs
w:          real matrix -> [O x m] matrix of network weights
max_it:     integer     -> Maximum number of iterations
alpha:      real number -> Learning rate
data:       real number -> Training set
validation: real number -> Validation set
```

**Output:**

```
SAÍDA:
    w: matriz real de pesos (o x m)
    b: vetor real de bias
```

# Pre-process

To start the neural network implementation, all the data in the file were read so that the class names were replaced by numbers from 1 to 3, where the number 1 indicates that the sample belongs to the species setosa, and the numbers 2 and 3 to the species virginica and versicolor, respectively. These data are then normalized by the zscore method and shuffled.

## Train / Validation / Test
Training: 70% of the dataset (105 samples)

Validation: 15% of the dataset (22 samples) 

Test: 15% of the dataset (23 samples)

The data are taken sequentially, since they are already shuffled, to compose the groups in the following order: the first samples will be for training, followed by testing, and finally validation.


# Training stage

To begin training, the training data is separated into two groups: one comprising only the net inputs, with a total of 4 inputs; and the other containing the column referring to which class each sample belongs to. For this group, since there are three possible outputs, species 1, 2 or 3, instead of treating it by any of these numbers, the data is transformed into a matrix of n rows (number of samples) x 3 columns. Each row will be filled with zeros, except for the column of its respective class, which will be assigned the value 1. That is, taking as an example that the k-th sample is from species 3, row k will be filled in the form: 0 0 1.

# Stopping Criteria / Activation Function
The stopping criteria for the training are: ```the number of epochs reaches the predefined maximum number of iterations, the quadratic error of the epoch is greater than zero, or if the quadratic error of the validation process starts to increase```.

 Softmax was used as the activation function. Different learning rates and initialization weights were tested.
As long as the stopping criterion is not met, the y-values of the network are calculated, and consequently the error value. With this, the new weights and the value of the bias are calculated. The bias is represented by a vector of size equal to the number of possible outputs from the net, in this case, 3. Each bias is added to the function in its respective neuron at the time y is calculated.

# Validation Step
Along with training, comes the validation step of the current epoch. As all training samples are viewed by the network, which in turn adjusts the w-value of the weights, the validation stage tests these values with its set of samples, resulting in a quadratic error value. The tendency of the error is to decrease, but if at any time this error starts to increase, it means that the network is tending to overfitting, so the training is stopped, and the neuron weights will be those with the lowest validation squared value.

# Test Step
After the process described above is completed, the capability of the trained network is tested with samples that have not yet been seen. For this, the network is fed with the values for w and bias, obtained from the test set as input.

# How to run
Run the script ```classification.m```

The script performs the training/validation/testing process with the following parameters:

- weights w: started with zero values
- learning rate = 0.3
- max_it = 300

And it provides the confusion matrix of each step (train/validation/test), network accuracy, weights and bias values.