#!/bin/bash -l
#SBATCH --job-name smallest-big-test
#SBATCH --output output_one_example.txt
#SBATCH --account "g101-2284"
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 24
#SBATCH --time 01:10:00

module load common/python/3.11
python3 run_tests.py
