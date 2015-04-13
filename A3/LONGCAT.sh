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

mkdir ./newmodel
cp /scratch/ls3470/DeepLearning/longcat/A3/newmodel/doall.lua ./newmodel
cp /scratch/ls3470/DeepLearning/longcat/A3/newmodel/load_glove.lua ./newmodel
cp /scratch/ls3470/DeepLearning/longcat/A3/newmodel/model_morefanout_bigger.lua ./newmodel
cp /scratch/ls3470/DeepLearning/longcat/A3/newmodel/preprocess_glove_plain_100.lua ./newmodel
cp /scratch/ls3470/DeepLearning/longcat/A3/newmodel/opt.lua ./newmodel
cp /scratch/ls3470/DeepLearning/longcat/A3/newmodel/test_model.lua ./newmodel
cp /scratch/ls3470/DeepLearning/longcat/A3/newmodel/train_model.lua ./newmodel

cp /scratch/ls3470/DeepLearning/longcat/A3/LONGCAT.pdf ./
