# ML-Assignment1
Python program to execute linear, weightedLinear, logistic regression and gaussian discriminant analysis.

An executable shell script with the name - run.sh.

## Running run.sh

Depending on the input command line arguments, the shell script should call/invoke the appropriate function/code and 
generate all the output for the respective question.

The first input argument is always the question number, second argument - relative or absolute path of the file containing 
x(features), third argument - the path of the file containing y and further arguments, if any depends on the question, 
e.g., for question 1, it is learning rate and time gap.

### Arguments for different questions:

#### Question 1: Linear Regression
./run.sh 1 <path_of_file_containing_x> <path_of_file_containing_y> <learning_rate> <time_gap_in_seconds>

./run.sh 1 linearX.csv linearY.csv 0.2 0.2

This should run the batch gradient descent with the provided learning rate, print the answer to part (a) and generate 
different plots corresponding to other parts.
 

#### Question 2: Weighted Linear Regression
./run.sh 2 <path_of_file_containing_x> <path_of_file_containing_y> <tau_>

/run.sh 2 weightedX.csv weightedY.csv 0.8

 
#### Question 3: Logistic Regression by Newton's Method
./run.sh 3 <path_of_file_containing_x> <path_of_file_containing_y>

 
#### Question 4: Gaussian Discriminant Analysis
./run.sh 4 <path_of_file_containing_x> <path_of_file_containing_y>
