#!/bin/bash

if [[ $1 =~ ^[+-]?[0-9]+\.?[0-9]*$ ]] ; then
 qsub relu.job.sh $1
 qsub mrelu.job.sh $1
 qsub mrelu-t.job.sh $1
 qsub prelu.job.sh $1
else
 echo "Must enter a learning rate!"
fi
