#!/bin/bash
#SBATCH -J deepcam-cgpu
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -c 64
#SBATCH --time 30

# Setup software environment
module load pytorch/v1.6.0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=32

# Job configuration
run_tag="deepcam_${SLURM_JOB_ID}"
data_dir_prefix="/global/cscratch1/sd/sfarrell/deepcam/data/n10-benchmark-data/data-replicated"
output_dir=$SCRATCH/deepcam/results/$run_tag

# Scale number of epochs according to the number of nodes
# Maybe use num_nodes / 8 to be comparable to cori-gpu..?
epochs=1 #$SLURM_JOB_NUM_NODES

# Create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

# Run training
srun -u --cpu_bind=cores \
     python ../train_hdf5_ddp.py \
     --wireup_method "mpi" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 0 \
     --optimizer "AdamW" \
     --start_lr 1e-3 \
     --validation_frequency 256 \
     --training_visualization_frequency 0 \
     --validation_visualization_frequency 0 \
     --logging_frequency 16 \
     --save_frequency 256 \
     --max_epochs $epochs \
     --local_batch_size 2 |& tee -a ${output_dir}/train.out
