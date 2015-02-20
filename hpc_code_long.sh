#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=10:00:00
#PBS -l mem=24GB
#PBS -m abe
#PBS -M ls3470@nyu.edu
#PBS -N doall

module load cuda/6.5.12
cd /home/ls3470/DeepLearning/A1

th doall.lua -save /scratch/ls3470/DeepLearning/A1/result -size small -type cuda
