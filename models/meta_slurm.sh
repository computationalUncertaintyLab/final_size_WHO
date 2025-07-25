#!/bin/bash

#SBATCH --partition=haswell,hawkcpu,rapids
 
#--Request 1 hour of computing time
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
 
#--Give a name to your job to aid in monitoring
#SBATCH --job-name runner
 
#--Write Standard Output and Error
#SBATCH --output="myjob.%j.%N.out"
 
cd ${SLURM_SUBMIT_DIR} # cd to directory where you submitted the job
bash ./models/distributor_inf.sh ./models/SLURM_runner.sh ./models/iteration_list.csv
exit
