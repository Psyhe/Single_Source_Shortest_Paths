#!/bin/bash
if [ -z ${OMPI_COMM_WORLD_LOCAL_RANK+x} ]; then RANK=$SLURM_PROCID; else RANK=$OMPI_COMM_WORLD_LOCAL_RANK; fi

./$1/sssp tests/$2/$RANK.in outputs/$RANK.out
