#!/usr/bin/env bash
#SBATCH -J null-4-mer-worker
#SBATCH -p condo
#SBATCH -q condo
#SBATCH -A ddp325
#SBATCH -c 1
#SBATCH -t 08:00:00
#SBATCH --mem=4GB
#SBATCH -a 1-100
#SBATCH --requeue

source $HOME/.bashrc
ulimit -s unlimited

# Resubmit if assigned to a node with a high failure rate
if [ $(grep $(hostname) $SLURM_SUBMIT_DIR/bad_nodes) ]; then

    sbsub $SLURM_SUBMIT_DIR/null-worker-condo.sbatch -a 1
    sleep 300
    exit
 
fi

# Setup local conda environment
CONDA_ENV_DIR=conda-env
PACKED_CONDA_ENV=openff-param-fit.tar.gz
COPY_DUMMY=copying-conda-env

LOCAL_SCRATCH_DIR=/scratch/ccavende/job_${SLURM_JOBID}
cd $LOCAL_SCRATCH_DIR

# Wait if another worker is copying the packed conda environment
while [ -f $COPY_DUMMY ]; do sleep 120; done

# Copy and unpack the packed conda environment
if [ ! -f $PACKED_CONDA_ENV ]; then

    touch $COPY_DUMMY
    mkdir -p $CONDA_ENV_DIR
    cp $HOME/protein-ff/param-fit/$PACKED_CONDA_ENV .
    tar -xf $PACKED_CONDA_ENV -C $CONDA_ENV_DIR
    rm $COPY_DUMMY

fi

# Activate the packed conda environment
source $CONDA_ENV_DIR/bin/activate

# Launch local worker
HOST="$(cat $SLURM_SUBMIT_DIR/host)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

work_queue_worker --cores=1 -s $LOCAL_SCRATCH_DIR -t 7200 \
    --disk-threshold=0.002 --disk==3000 --memory=3000 --gpus=0 $HOST 55125

