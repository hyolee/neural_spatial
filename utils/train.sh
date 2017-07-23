#!/bin/bash

dir=$1
prefix=$2

net=/om/user/hyo/metrics/neural_spatial/"$dir"/train_val_"$prefix".prototxt #"/om/user/hyo/metrics/neural_spatial/test_loss3_3/train_val_alexnet_small_lw""$1"".prototxt"
test_interval=5000000
base_lr=0.01
lr_policy="step"
gamma=0.1
stepsize=100000
max_iter=450000
momentum=0.9
iter_size=1
weight_decay=0.0005
snapshot=1000
snapshot_prefix=/om/user/hyo/metrics/neural_spatial/"$dir"/snapshot/"$prefix" #"/om/user/hyo/metrics/neural_spatial/test_loss3_3/snapshot/alexnet_small_lw""$1"
iter_resume=$3

dp_params="{'data_path':'/om/user/yamins/.skdata/imagenet/ChallengeSynsets2013_offline_23d6ee636ade8ad3912204410d1acc23c10357cf/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/data.raw','data_key':'data','label_path':'/om/user/hyo/latent/googlenet2/data/label_inet.h5','label_key':['nc'],'cache_type':'hdf5','batch_size':256,'val_len':50000}"
preproc="{'data_mean':'/om/user/hyo/caffe/imagenet_mean.npy','crop_size':227,'do_img_flip':True,'noise_level':10}"

if [ -z "$iter_resume" ]
then
  python /om/user/hyo/metrics/neural_spatial/utils/train.py -n $net -l $base_lr -c $lr_policy -g $gamma -z $stepsize -e $max_iter -m $momentum -i $iter_size -w $weight_decay -s $snapshot -p $snapshot_prefix -t $test_interval --dp-params=$dp_params --preproc=$preproc --do-test=0
else
  python /om/user/hyo/metrics/neural_spatial/utils/train.py -n $net -l $base_lr -c $lr_policy -g $gamma -z $stepsize -e $max_iter -m $momentum -i $iter_size -w $weight_decay -s $snapshot -p $snapshot_prefix -t $test_interval -r $iter_resume --dp-params=$dp_params --preproc=$preproc --do-test=0
fi
