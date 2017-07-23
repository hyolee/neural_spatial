#!/bin/bash

dir=$1
prefix=$2
#lw=$2
#name="SOMETHING" #"alexnet_small_lw""$2"
iter_resume=$3
gpu_spec=$4

if [ "$gpu_spec" == "titan-x" ]
then
  gpu_spec="$gpu_spec"":"
else
  gpu_spec=""
fi

name=$prefix

if [ -z "$iter_resume" ]
then
  sbatch --job-name=$name --mem=5000 --gres=gpu:"$gpu_spec"1 --time=07-00 --out=/om/user/hyo/metrics/neural_spatial/"$dir"/output/"$name".out --error=/om/user/hyo/metrics/neural_spatial/"$dir"/output/"$name".out /om/user/hyo/metrics/neural_spatial/utils/train.sh $dir $prefix
else
  sbatch --job-name=$name --mem=5000 --gres=gpu:"$gpu_spec"1 --time=07-00 --out=/om/user/hyo/metrics/neural_spatial/"$dir"/output/"$name"_r"$iter_resume".out --error=/om/user/hyo/metrics/neural_spatial/"$dir"/output/"$name"_r"$iter_resume".out /om/user/hyo/metrics/neural_spatial/utils/train.sh $dir $prefix $iter_resume 
fi

