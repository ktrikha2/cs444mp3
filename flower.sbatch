#!/bin/bash

### Do not change the following 8 lines.
#SBATCH --job-name=flower_demo
#SBATCH --account=bfdu-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=48g
### End Do not change block

#SBATCH --mail-user=NETID@illinois.edu  # replace with your email
#SBATCH --mail-type=NONE                # Type of email notifications to send

#SBATCH --output=ece494mp3.1-%x-%j.out  # Default name of batch job output file if not set on command line
#SBATCH --error=ece494mp3.1-%x-%j.err   # Default name of batch job error file if not set on command line

# This creates output directory for every run. Change EXP_NAME and DIR_NAME
# appropriately while performing each run and ablation experiment

# Default values for DIR_NAME and EXP_NAME. For each different experiment you
# run, you should use a different EXP_NAME. Change it in the command line flag.
DIR_NAME=${DIR_NAME:?"DIR_NAME not set"}
EXP_NAME=${EXP_NAME:?"EXP_NAME not set"}

mkdir -p ${DIR_NAME}
LOG_FILE="${DIR_NAME}/log.txt"

module load python/3.10.14 cuda/12.4.0 gcc-native/12.3

# Activate your venv
source /work/nvme/bfdu/cs444/fa25/mp3/venv/bin/activate

# print some diagnostics
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
export PYTHONUNBUFFERED=1

# Run your job 

python demo_flower.py --exp_name $EXP_NAME \
    --output_dir ${DIR_NAME} \
    --data_dir /work/nvme/bfdu/cs444/fa25/mp3/data/flower-dataset-reduced/ 2>&1 | tee ${LOG_FILE}
