#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=6:00:00
#PBS -l mem=1GB
#PBS -N LONGCAT

mkdir ~/LONGCAT
cd ~/LONGCAT

cp /scratch/ls3470/DeepLearning/longcat/A3/baseline/A3_skeleton.lua ./
cp /scratch/ls3470/DeepLearning/longcat/A3/baseline/A3_baseline.lua ./
cp /scratch/ls3470/DeepLearning/longcat/A3/LONGCAT_model.sh ./
cp /scratch/ls3470/DeepLearning/longcat/A3/run_model_on_stdin.lua ./

cp /scratch/ls3470/DeepLearning/longcat/A3/LONGCAT.pdf ./
cp /scratch/ls3470/DeepLearning/longcat/A3/model.net ./
