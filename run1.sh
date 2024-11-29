#!/bin/sh
### General options

### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J influence_matrix
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues
#BSUB -W 24:00
### -- request system memory --
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
#BSUB -u s206182@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/12.0
source /work3/s206182/venv/py_chest/bin/activate

# Run the Python script
python3 ./IM.py
