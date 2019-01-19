#!/bin/bash

experiment_name="mnist_gif"
input_dir="./logs/${experiment_name}"
output_dir="./logs"
convert -layers optimize -loop 0 -delay 10 ${input_dir}/*.png ${output_dir}/${experiment_name}.gif


