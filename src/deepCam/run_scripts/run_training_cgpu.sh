#!/bin/bash
#SBATCH -J deepcam-cgpu
#SBATCH -C gpu
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time-min 4:00:00
#SBATCH --time 24:00:00

# Setup software environment
module load cgpu
module load pytorch/v1.6.0-gpu
#conda activate mlperf_deepcam
#module load pytorch/v1.4.0
#export PROJ_LIB=/global/homes/t/tkurth/.conda/envs/mlperf_deepcam/share/basemap
#export PYTHONPATH=/global/homes/t/tkurth/.conda/envs/mlperf_deepcam/lib/python3.7/site-packages:${PYTHONPATH}

# Job configuration
rankspernode=8
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))
run_tag="deepcam_004"
data_dir_prefix="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist"
output_dir=$SCRATCH/deepcam/results/$run_tag

# Create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

# Run training
srun -u -N ${SLURM_NNODES} -n ${totalranks} -c $(( 80 / ${rankspernode} )) --cpu_bind=cores \
     python ../train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 0 \
     --model_prefix "classifier" \
     --optimizer "LAMB" \
     --start_lr 1e-4 \
     --lr_schedule type="multistep",milestones="8192 16384",decay_rate="0.1" \
     --lr_warmup_steps 1024 \
     --lr_warmup_factor 16. \
     --weight_decay 1e-2 \
     --validation_frequency 200 \
     --training_visualization_frequency 0 \
     --validation_visualization_frequency 0 \
     --logging_frequency 10 \
     --save_frequency 400 \
     --max_epochs 200 \
     --amp_opt_level O1 \
     --enable_wandb \
     --wandb_certdir $HOME \
     --local_batch_size 2 |& tee -a ${output_dir}/train.out
     #--max_validation_steps 50 \
