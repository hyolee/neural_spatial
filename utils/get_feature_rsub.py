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

print "Extracting " + layer + " from " + sys.argv[2] + "..."
N = Net(train_val, snapshot, dp_params, preproc, test=True)

dshape = N.N.blobs[layer].data.shape
nfeat = np.prod(dshape[1:])
dim_grid = dshape[2:]

# randomly sample features
nSample = 5000
random_seed = 0
np.random.seed(random_seed)
inds = np.sort(np.random.randint(nfeat, size=nSample)) # randomly select
ind_coords = [(ind/np.prod(dim_grid), (ind%np.prod(dim_grid))/dim_grid[1], (ind%np.prod(dim_grid))%dim_grid[1]) for ind in inds] # convert indices into coordinates
nfs = np.array(ind_coords)[:,0]
xs = np.array(ind_coords)[:,1]
ys = np.array(ind_coords)[:,2]

print "Saving features to", output_file, "..."
f = h5py.File(output_file, 'w')
N.forward()
Fbn = np.copy(N.N.blobs[layer].data)
F = Fbn[:,nfs,xs,ys]
for id in range(22):
    N.forward()
    Fbn = np.copy(N.N.blobs[layer].data)
    F = np.append(F, Fbn[:,nfs,xs,ys], axis=0)
F = F[:5760] # pick only 5760 images (# of total HvM images)
f['data'] = F
f['ind_coords'] = ind_coords
f.close()

