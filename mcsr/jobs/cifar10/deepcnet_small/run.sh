#!/bin/bash

if [[ $1 =~ ^[+-]?[0-9]+\.?[0-9]*$ ]] ; then
 qsub -v lr=$1 relu.job.sh
 qsub -v lr=$1 mrelu.job.sh
 qsub -v lr=$1 mrelubias.job.sh
 qsub -v lr=$1 mrelubias-t.job.sh
 qsub -v lr=$1 mrelu-t.job.sh
 qsub -v lr=$1 prelu.job.sh
else
 echo "Must enter a learning rate!"
fi
