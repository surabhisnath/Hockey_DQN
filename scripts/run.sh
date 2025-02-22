#!/bin/bash
#SBATCH --job-name=hockey # give it any name you want
#SBATCH --cpus-per-task=4 # max 24 per node
#SBATCH --partition=day
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=csahiti07@gmail.com

singularity run /home/stud460/container.sif python3 ./scripts/runner.py --env hockey --alpha 0.0002 --epsilon 0.2 --epsilondecay 1.0 --numepisodes 15000 --numseeds 1 --opponent self --selfplayfilename agent_hockey_0_1.pth  
