#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=1:00:00
#PBS -l mem=2GB
#PBS -M cao324@nyu.edu
#PBS -m abe
#PBS -j oe
#PBS -N longcat

module load cuda/6.5.12

gitdir="/home/$USER/longcat"

if [ ! -d $gitdir ]; then
    cd "/home/$USER/"
    git clone https://github.com/catherio/longcat.git
else
    echo "Code directory already exists, not cloning"
fi

cd ${gitdir}/A2

th 1_dataGet.lua -datatransfer hpc
