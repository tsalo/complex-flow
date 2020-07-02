#!/bin/bash
#---Number of cores
#SBATCH -c 4

#---Job's name in SLURM system
#SBATCH -J dwi

#---Error file
#SBATCH -e dwi_err

#---Output file
#SBATCH -o dwi_out

#---Queue name
#SBATCH --account iacc_nbc

#---Partition
#SBATCH -p centos7
########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=$NPROCS
. $MODULESHOME/../global/profile.modules
module load singularity-3

source /home/data/nbc/nbclab-env/py3_environment
python test_dwidenoise.py
