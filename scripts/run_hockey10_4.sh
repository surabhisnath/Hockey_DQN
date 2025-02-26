#!/bin/bash
#SBATCH --job-name=run_hockey  # give it any name you want
#SBATCH --ntasks=1                                                                             # Number of tasks (see below)
#SBATCH --cpus-per-task=8                                                                     # Number of CPU cores per task
#SBATCH --nodes=1                                                                              # Ensure that all cores are on one machine
#SBATCH --time=12:00:00                                                                         # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                                                                           # Request 1 GPU
#SBATCH --mem=64G                                                                              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --error=job.%J.err  # %J is the job ID, errors will be written to this file
#SBATCH --output=job.%J.out # the output will be written in this file
#SBATCH --mail-type=ALL                                                                   # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=surabhi.s.nath@gmail.com                                        # Email to which notifications will be sent
#SBATCH --partition=a100-galvani

# print info about current job
scontrol show job $SLURM_JOB_ID

source /home/dayan/dno388/.bashrc
conda activate /mnt/lustre/work/dayan/dno388/conda_envs/RLproject

python /mnt/lustre/work/dayan/dno388/RL_project/scripts/runner.py --env hockey --numepisodes 10000 --agentfilename agent_hockey_0_10.pth --opponent weak --trainmode defense --numseeds 1 --epsilon 1  --minepsilon 0.2 --epsilondecay 0.9998 --savenum 10_4 --per --dueling