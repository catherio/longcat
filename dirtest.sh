#!/bin/bash
DIRNAME=$(date +"%m-%d_%H:%M")_results
echo $DIRNAME
if [ -d "$DIRNAME" ]; then
    mkdir $DIRNAME
fi
th doall.lua -save $DIRNAME -size small
