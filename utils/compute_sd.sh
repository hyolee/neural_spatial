#!/bin/bash

# alexnet, inet, cat
feature_path=/om/user/hyo/metrics/neural_spatial/dimensionality/features/F_alexnet_inet_iter440000_conv1_rsub.h5
coord_path=/om/user/hyo/metrics/neural_spatial/dimensionality/coord/coord_alexnet_inet_conv1.h5
iter="random"
tissue_size="30."

python compute_sd.py $feature_path $coord_path $iter $tissue_size
