#!/bin/bash

module load cuda/6.5.12
module load torch

cd ~/LONGCAT

/home/cao324/torch/install/bin/th run_model_on_stdin.lua
