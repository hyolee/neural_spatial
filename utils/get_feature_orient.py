debug_root="/om/user/hyo/metrics/neural_spatial/test_loss4/"
import sys
sys.path.insert(0, debug_root)
from debug import *
import h5py

train_val = sys.argv[1]
snapshot = sys.argv[2]
output_file = sys.argv[3]
layer = sys.argv[4]	
dp_params = {'data_path':'/om/user/hyo/metrics/neural_spatial/utils/orientation_stims.h5','data_key':'data','label_path':'/om/user/hyo/metrics/neural_spatial/utils/orientation_stims.h5','label_key':['nc'],'cache_type':'hdf5','batch_size':256,'val_len':1440}

print "Extracting " + layer + " for orientation stims..."

N = Net(train_val, snapshot, dp_params, preproc, test=True)

#N.forward()
#F = N.N.blobs[layer].data
#for i in range(22):
#    N.forward()
#    F = np.append(F, N.N.blobs[layer].data, axis=0)
#F = F[:5760] # pick only 5760 images (# of total HvM images)
#
#if '.p' in output_file:
#    with open(output_file, "w") as f:
#        cPickle.dump(F, f)
#elif '.h5' in output_file:
#    f = h5py.File(output_file, 'w')
#    for nfil in range(F.shape[1]):
#        F_nfil = F[:,nfil,:,:]
#        f[str(nfil)] = F_nfil
#    f.close()

if '.p' in output_file:
    N.forward()
    F = np.copy(N.N.blobs[layer].data)
    for i in range(22):
        N.forward()
        F = np.append(F, np.copy(N.N.blobs[layer].data), axis=0)
    F = F[:1440]
    with open(output_file, "w") as f:
        cPickle.dump(F, f)
elif '.h5' in output_file:
    N.forward()
    f = h5py.File(output_file, 'w')
    for i in range(23):
        N.forward()
        f['bn'+str(i)] = np.copy(N.N.blobs[layer].data)
    f.close()

