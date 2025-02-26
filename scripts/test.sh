#!/bin/bash
#SBATCH --job-name=test_hockey  # give it any name you want
#SBATCH --cpus-per-task=4   # max 24 per node
#SBATCH --partition=day     # choose out of day, week, month depending on job duration
#SBATCH --mem-per-cpu=3G    # max 251GB per node
#SBATCH --gres=gpu:1        # how many gpus to use each node has 4 gpus
#SBATCH --time=08:00:00     # job length: the job will run either until completion or until this timer runs out
#SBATCH --error=job.%J.err  # %J is the job ID, errors will be written to this file
#SBATCH --output=job.%J.out # the output will be written in this file
#SBATCH --mail-type=ALL     # write a mail if a job begins, ends, fails, gets requeued or stages out options: NONE, BE>
#SBATCH --mail-user=csahiti07@gmail.com   # your email

<<<<<<< HEAD
# print info about current job
scontrol show job $SLURM_JOB_ID

source /home/dayan/dno388/.bashrc
conda activate /mnt/lustre/work/dayan/dno388/conda_envs/RLproject
python /mnt/lustre/work/dayan/dno388/RL_project/scripts/test.py --filename agent_hockey_0_1_20000.pth --savegif
=======
singularity run ../../container.sif python3 test.py --filename agent_hockey_0_154.pth --numtestepisodes 100 --opponent weak
>>>>>>> 521e97aa36a56e259d7e08b4e0d5ed58a7fa3f39
