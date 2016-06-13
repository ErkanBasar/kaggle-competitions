#!/bin/bash
#SBATCH -n 10
#SBATCH -t 100:00:00

module load python/2.7.11
python get_feature_classif.py
echo "finished"
