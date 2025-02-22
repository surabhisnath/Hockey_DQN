#!/bin/bash
#SBATCH --job-name=test_hockey  # give it any name you want
#SBATCH --ntasks=1                                                                             # Number of tasks (see below)
#SBATCH --cpus-per-task=8                                                                     # Number of CPU cores per task
#SBATCH --nodes=1                                                                              # Ensure that all cores are on one machine
#SBATCH --time=12:00:00                                                                         # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                                                                           # Request 1 GPU
#SBATCH --mem=64G                                                                              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --error=job.%J.err  # %J is the job ID, errors will be written to this file
#SBATCH --output=job.%J.out # the output will be written in this file
#SBATCH --mail-type=ALL                                                                   # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=csahiti07@gmail.com                                        # Email to which notifications will be sent


singularity run ../../container.sif python3 test.py --filename agent_hockey_0_20.pth --numtestepisodes 100 --opponent weak
