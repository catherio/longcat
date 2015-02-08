#!/bin/bash
PARAMS=(2 5 7)
N=${#PARAMS[@]}
for i in `seq 0 $((N-1))`;
do
	PARAM=${PARAMS[$i]}
	th cmdtest.lua -param $PARAM
done

