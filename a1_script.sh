#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l walltime=10:00:00
#PBS -l mem=24GB
#PBS -m abe
#PBS -M cao324@nyu.edu
#PBS -N doall

module load cuda/6.5.12
cd /scratch/cao324/longcat/A1/

PARAMS=(.3 .35 .4 1)
N=${#PARAMS[@]}
for i in `seq 0 $((N-1))`;
do
	PARAM=${PARAMS[$i]}
	echo $PARAM

	DIRNAME=$(date +"%m-%d_%H%M")_results_gausswin$PARAM
	if [ -d "$DIRNAME" ]; then
    		mkdir $DIRNAME
	fi	
	
	th doall.lua -save $DIRNAME -size small -type cuda -gausswidth $PARAM
done
