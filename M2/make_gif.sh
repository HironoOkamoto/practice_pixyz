#!/bin/bash

list=("categorical_samples" "gumbel_softmax_samples_0.1" "gumbel_softmax_samples_0.5" "gumbel_softmax_samples_1" "gumbel_softmax_samples_10" "gumbel_softmax_samples_100")
for l in ${list[@]}
do
    input_dir="./logs/${l}"
    output_dir="./logs"
    convert -layers optimize -loop 0 -delay 300 ${input_dir}/*.png ${output_dir}/${l}.gif
done

