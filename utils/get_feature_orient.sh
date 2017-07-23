#!/bin/bash

dir=$1
prefix=$2
iter=$3
layer=$4
data="orient"

net=/om/user/hyo/metrics/neural_spatial/"$dir"/train_val_"$prefix".prototxt
snapshot=/om/user/hyo/metrics/neural_spatial/"$dir"/snapshot/"$prefix"_iter_"$iter".caffemodel
output_file=/om/user/hyo/metrics/neural_spatial/"$dir"/features/F_"$prefix"_"$data"_iter"$iter".h5

python /om/user/hyo/metrics/neural_spatial/utils/get_feature_orient.py $net $snapshot $output_file $layer
