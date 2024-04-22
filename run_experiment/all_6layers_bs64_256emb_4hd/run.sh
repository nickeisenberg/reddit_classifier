#! /usr/bin/bash

#SBATCH --job-name=6ly_bs64_256emb_4h
#SBATCH --output=out.%x.o%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --partition=pbatch
#SBATCH --time=12:00:00

srun /g/g11/eisenbnt/venvs/base/bin/python3 \
    -u /g/g11/eisenbnt/GitRepos/reddit_classifier/run_experiment/all_6layers_bs64_256emb_4hd/__main__.py

