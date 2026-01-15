#!/bin/bash
#SBATCH -J b7s26-sage-4-mer-driver
#SBATCH -p platinum
#SBATCH -q hcp-csd765
#SBATCH -A ddp325
#SBATCH -c 2
#SBATCH -t 7-00:00:00
#SBATCH --mem=4GB

source $HOME/.bashrc
ulimit -s unlimited
conda activate openff-param-fit

FB_DIR=$SLURM_SUBMIT_DIR/forcebalance
LOCAL_SCRATCH_DIR=/scratch/ccavende/job_${SLURM_JOBID}

rsync -avzIi $FB_DIR/optimize.in $LOCAL_SCRATCH_DIR
rsync -avzIi $FB_DIR/targets.tar.gz $LOCAL_SCRATCH_DIR
rsync -avzIi $FB_DIR/forcefield $LOCAL_SCRATCH_DIR

cd $LOCAL_SCRATCH_DIR

tar -zxvf targets.tar.gz

echo $(hostname) > $SLURM_SUBMIT_DIR/host

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if ForceBalance.py optimize.in ; then

    rsync -avzIi --exclude="optimize.tmp" --exclude="optimize.bak" \
        --exclude="targets" --exclude=wq* $LOCAL_SCRATCH_DIR/* $FB_DIR/ \
        > $SLURM_SUBMIT_DIR/rsync_log

fi

echo 'Completed optimization for b7s26 driver'

