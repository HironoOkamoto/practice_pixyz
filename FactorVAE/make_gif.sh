#!/bin/bash

list=("0" "1" "2")
for l in ${list[@]}
do
    experiment_name="betavae_C_dsprites_z_dim10_gamma80_gif_${l}"
    input_dir="./logs/${experiment_name}"
    output_dir="./logs"
    convert -layers optimize -loop 0 -delay 20 ${input_dir}/*.png ${output_dir}/${experiment_name}.gif
done

