#!/bin/bash
#SBATCH -J deepcam-cgpu
#SBATCH -C gpu
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 1:30:00

# Setup software environment
module load cgpu
module load pytorch/v1.6.0-gpu

# Job configuration
rankspernode=8
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))
run_tag="deepcam_${SLURM_JOB_ID}"
output_dir=/tmp/deepcam/results/$run_tag

# 480 GB
data_dir_prefix="/global/cscratch1/sd/sfarrell/deepcam/data/n10-benchmark-data/data-replicated"
# 66 GB
#data_dir_prefix="/global/cscratch1/sd/sfarrell/deepcam/data/dry-run"

# Scale number of epochs according to the number of nodes
epochs=$SLURM_JOB_NUM_NODES

# Create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

# Copy from VAST (must be already populated) which may be faster than cscratch.
VAST_DIR="/vast/$USER/deepcam_$(basename $data_dir_prefix)"
NVME_DIR="/tmp/${USER}_deepcam_$(basename $data_dir_prefix)"
time rsync -aL "$VAST_DIR/" "$NVME_DIR/"

data_dir_prefix="$NVME_DIR"
echo "Input dir is '$data_dir_prefix'; content of dir:"; ls -l "$data_dir_prefix"

# Run training
srun -u --cpu_bind=cores \
     python ../train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 2 \
     --optimizer "AdamW" \
     --start_lr 1e-3 \
     --validation_frequency 256 \
     --training_visualization_frequency 0 \
     --validation_visualization_frequency 0 \
     --logging_frequency 16 \
     --save_frequency 256 \
     --max_epochs $epochs \
     --enable_amp \
     --local_batch_size 2 |& tee -a ${output_dir}/train.out
