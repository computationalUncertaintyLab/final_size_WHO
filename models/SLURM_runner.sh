#!/bin/bash

#SBATCH --partition=hawkcpu
 
#--Request 1 hour of computing time
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
 
#--Give a name to your job to aid in monitoring
#SBATCH --job-name flumodel
 
#--Write Standard Output and Error
#SBATCH --output="myjob.%j.%N.out"

cd ${SLURM_SUBMIT_DIR} # cd to directory where you submitted the job
 
#--export environmental variables
export LOCATION=${LOCATION}
export SEASON=${SEASON}

echo ${LOCATION} - ${SEASON}
pwd


.test/bin/python ./models/train_past_model_outputs.py --LOCATION ${LOCATION} --SEASON ${SEASON}

echo "Python script completed with exit code $?"
echo "complete"

exit
