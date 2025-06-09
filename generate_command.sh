#!/bin/bash
# Usage: ./generate_command.sh test_folder_name
# This script is called by each rank with SLURM_PROCID or OMPI_COMM_WORLD_LOCAL_RANK

if [ -z ${OMPI_COMM_WORLD_LOCAL_RANK+x} ]; then RANK=$SLURM_PROCID; else RANK=$OMPI_COMM_WORLD_LOCAL_RANK; fi

./dijkstra_compiled new_tests/$1/$RANK.in new_tests/$1/$RANK.out