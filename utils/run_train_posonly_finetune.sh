#!/bin/bash

dir=$1
prefix=$2
#lw=$2
#name="SOMETHING" #"alexnet_small_lw""$2"
learned_weights=$3
base_lr=$4
gpu_spec=$5

if [ "$gpu_spec" == "titan-x" ]
then
  gpu_spec="$gpu_spec"":"
else
  gpu_spec=""
fi

if [ -z "$base_lr" ]
then
  name="$prefix"
else
  name="$prefix""_lr""$base_lr"
fi

sbatch --job-name=$prefix --mem=5000 --gres=gpu:"$gpu_spec"1 --time=07-00 --out=/om/user/hyo/metrics/neural_spatial/"$dir"/output/"$name".out --error=/om/user/hyo/metrics/neural_spatial/"$dir"/output/"$name".out /om/user/hyo/metrics/neural_spatial/utils/train_posonly_finetune.sh $dir $prefix $learned_weights $base_lr
