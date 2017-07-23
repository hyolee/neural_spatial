#!/bin/bash

# examples of executing get_feature.py

# alexnet, inet, cat-trained
train_val="/om/user/hyo/metrics/neural_spatial/dimensionality/train_val/train_val_alexnet.prototxt"
snapshot="/om/user/hyo/metrics/neural_spatial/dimensionality/snapshot/alexnet_inet_iter_440000.caffemodel"
layer="conv1"
output_file="/om/user/hyo/metrics/neural_spatial/dimensionality/features/F_alexnet_inet_iter440000_""$layer""_rsub.h5"
data="inet"
python get_feature_rsub.py $train_val $snapshot $output_file $layer $data
