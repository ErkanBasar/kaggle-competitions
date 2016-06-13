#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 100:00:00

module load python/2.7.11 opencv/gnu
python feature_extraction.py
echo "finished"
