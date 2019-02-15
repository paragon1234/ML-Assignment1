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

#### Question 1
./run.sh 1 <path_of_file_containing_x> <path_of_file_containing_y> <learning rate> <time_gap_in_seconds>

This should run the batch gradient descent with the provided learning rate, print the answer to part (a) and generate 
different plots corresponding to other parts.
 

#### Question 2
./run.sh 2 <path_of_file_containing_x> <path_of_file_containing_y> <tau>

 
#### Question 3
./run.sh 3 <path_of_file_containing_x> <path_of_file_containing_y>

 
#### Question 4
./run.sh 4 <path_of_file_containing_x> <path_of_file_containing_y>
