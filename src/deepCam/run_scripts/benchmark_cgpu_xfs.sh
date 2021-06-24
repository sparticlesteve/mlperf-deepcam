#!/bin/bash
#SBATCH -J deepcam-cgpu
#SBATCH -C gpu
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 2:00:00

#SBATCH --image=nersc/pytorch:ngc-21.02-v0
#SBATCH --volume="/global/cscratch1/sd/chiusole/tmpfiles:/tmp_xfs:perNodeCache=size=500G"

# Setup software environment
module load cgpu
module load pytorch/v1.6.0-gpu

# Job configuration
rankspernode=8
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))
run_tag="deepcam_${SLURM_JOB_ID}"

# 480 GB
#data_dir_prefix="/global/cscratch1/sd/sfarrell/deepcam/data/n10-benchmark-data/data-replicated"
# 66 GB
data_dir_prefix="/global/cscratch1/sd/sfarrell/deepcam/data/dry-run"

# Scale number of epochs according to the number of nodes
epochs=$SLURM_JOB_NUM_NODES

data_dir_prefix="/vast/$USER/deepcam_$(basename $data_dir_prefix)/"
XFS_DIR="/tmp_xfs/deepcam_data"
output_dir="/tmp_xfs/deepcam/results/$run_tag"

srun -u --cpu_bind=cores \
  shifter --env HDF5_USE_FILE_LOCKING=FALSE <<EOF

    # Use plain old cp in containers for portability. But this will take ages
    # because all N procs will copy the same data, causing a lot of
    # contention. I couldn't find a way to restrict this copy to a single
    # proc and put the others waiting on a barrier, so I ended up dropping
    # tests of deepcam on XFS.
    echo "Input dir is '$XFS_DIR'"
    time cp -a $data_dir_prefix/ $XFS_DIR/
    echo "Content of dir:"; ls -l "$XFS_DIR"; echo "size:"; du -sh "$XFS_DIR/"

    mkdir -p ${output_dir}
    touch ${output_dir}/train.out

     python ../train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${XFS_DIR} \
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

EOF
