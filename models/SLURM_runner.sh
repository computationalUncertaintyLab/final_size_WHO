#!/bin/bash

#SBATCH --partition=haswell,hawkcpu,rapids
 
#--Request 1 hour of computing time
#SBATCH --time=07:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
 
#--Give a name to your job to aid in monitoring
#SBATCH --job-name flumodel
 
#--Write Standard Output and Error
#SBATCH --output="myjob.%j.%N.out"
 
cd ${SLURM_SUBMIT_DIR} # cd to directory where you submitted the job
 
#--launch job

python -V

#--export environmental variables
export LOCATION=${LOCATION}
export TIME=${TIME}

echo ${LOCATION} - ${TIME}

.whoseason/bin/python3 ./models/train_past_model_outputs.py --LOCATION ${LOCATION} --SEASON ${SEASON}
 
exit
