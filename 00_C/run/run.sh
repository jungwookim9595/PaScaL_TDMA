#!/bin/bash
#SBATCH -J test_kdh
#SBATCH -p batch
#SBATCH -w cpu06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH -o results/%x_%j.out
#SBATCH -e results/%x_%j.err
#SBATCH --comment xxx

module purge 
module load nvhpc/23.7
# module load vtune/latest

mpirun -np 16 ./a.out ./PARA_INPUT.txt
# srun -n 8 ./a.out ./PARA_INPUT.txt
# mpirun -np 64 -gtool "vtune -collect hpc-performance -data-limit=0 -r vtune_result:1" ./a.out ./PARA_INPUT.txt

