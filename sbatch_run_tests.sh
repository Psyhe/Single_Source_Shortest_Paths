#!/bin/bash -l
#SBATCH --job-name 25_again_big_test_delta
#SBATCH --output output_delta_25_again.txt
#SBATCH --account "g101-2284"
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 24
#SBATCH --time 01:10:00

module load common/python/3.11
python3 run_tests.py
