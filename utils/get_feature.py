from debug import *
import numpy as np
import h5py

train_val = sys.argv[1]
snapshot = sys.argv[2]
output_file = sys.argv[3]
layer = sys.argv[4]	
if sys.argv[5] == 'inet':
    dp_params = {'data_path':'/om/user/yamins/.skdata/imagenet/ChallengeSynsets2013_offline_23d6ee636ade8ad3912204410d1acc23c10357cf/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/data.raw','data_key':'data','label_path':'/om/user/hyo/latent/googlenet2/data/label_inet.h5','label_key':['nc'],'cache_type':'hdf5','batch_size':256,'val_len':50000}
elif sys.argv[5] == 'hvm':
    dp_params = {'data_path':'/om/user/hyo/.skdata/HvMWithDiscfade_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/data.raw','data_key':'data','label_path':'/om/user/hyo/metrics/neural_spatial/dimensionality/features/label_hvm.h5','label_key':['nc'],'cache_type':'hdf5','batch_size':256,'val_len':5760}
else:
    print "Wrong test data!!" 

print "Extracting " + layer + " from " + sys.argv[5] + "..."

N = Net(train_val, snapshot, dp_params, preproc, test=True)

if '.p' in output_file:
    N.forward()
    F = np.copy(N.N.blobs[layer].data)
    for i in range(22):
        N.forward()
        F = np.append(F, np.copy(N.N.blobs[layer].data), axis=0)
    F = F[:5760] # pick only 5760 images (# of total HvM images)
    with open(output_file, "w") as f:
        cPickle.dump(F, f)
elif '.h5' in output_file:
    N.forward()
    f = h5py.File(output_file, 'w')
    for i in range(23):
        N.forward()
        f['bn'+str(i)] = np.copy(N.N.blobs[layer].data)
    f.close()

