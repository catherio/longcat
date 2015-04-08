#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=1:00:00
#PBS -l mem=24GB
#PBS -M cao324@nyu.edu
#PBS -m abe
#PBS -N srgen

module load cuda/6.5.12
cd /scratch/cao324/longcat/A2

#th 7_surrogateGen.lua

echo 'hi'