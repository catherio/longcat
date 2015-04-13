#!/bin/bash

# ADD ANYTHING ELSE THAT MIGHT NEED TO add torch/lua, our code etc to the path
module load torch

cd /home/kab695/DL/longcat/A3

th run_model_on_stdin.lua
