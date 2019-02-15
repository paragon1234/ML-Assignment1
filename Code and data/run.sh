#!/bin/sh

if [ $1 -eq 1 ]
then
 ./linearRegression.py $2 $3 $4 $5
elif [ $1 -eq 2 ]
then
 ./weightedLinearRegression.py $2 $3 $4
elif [ $1 -eq 3 ]
then
 ./logisticRegression.py $2 $3
else
 ./GDA.py $2 $3 $4
fi

   

