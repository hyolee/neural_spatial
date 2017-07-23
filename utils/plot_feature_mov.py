import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../utils/')
from debug import *
from plot_util import plot_dist
import numpy as np

dir = sys.argv[1]
prefix = sys.argv[2] #"_v2_posonly_fc6"  #"" #"_1layer" #sys.arg[1]
iters = sys.argv[3:]
nfs = None

cs = None
pds = None

for iter in iters:
    net = '/om/user/hyo/metrics/neural_spatial/' + dir +'train_val_alexnet' + prefix + '.prototxt'
    #snapshot = '/om/user/hyo/metrics/neural_spatial/dimensionality/snapshot/alexnet_inet_iter_440000.caffemodel'
    snapshot = '/om/user/hyo/metrics/neural_spatial/' + dir + '/snapshot/alexnet' + prefix + '_iter_' + str(iter) + '.caffemodel'
    #snapshot = '/om/user/hyo/metrics/neural_spatial/test_loss/bvlc_alexnet.caffemodel'
    #snapshot_prefix = 'snapshot/alexnet_' + prefix + '_iter_'
    dp_params = {'data_path':'/om/user/yamins/.skdata/imagenet/ChallengeSynsets2013_offline_23d6ee636ade8ad3912204410d1acc23c10357cf/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/data.raw','data_key':'data','label_path':'/om/user/hyo/latent/googlenet2/data/label_inet.h5','label_key':['nc'],'cache_type':'hdf5','batch_size':256,'val_len':50000}
    preproc = {'data_mean':'/om/user/hyo/caffe/imagenet_mean.npy','crop_size':227,'do_img_flip':True,'noise_level':10}
    
    N = Net(net, snapshot, dp_params, preproc, test=True)
    N.forward()
    
    data = N.N.blobs['fc6_raw'].data[:]
    posx = np.array(N.N.params['loss_scl'][0].data[:])
    posy = np.array(N.N.params['loss_scl'][1].data[:])
    if len(posx.shape)==1:
      posx = posx.reshape(len(posx), 1)
      posy = posy.reshape(len(posy), 1)
    
    # compute corr
    m = np.mean(data, axis=0)
    numer = data - m
    denom = np.sqrt(np.sum((data-m)**2, axis=0))
    Cs = (data - m) / denom
    C = np.dot(np.transpose(Cs), Cs)
    dim = C.shape[0]
    # compute dist
    PD = np.sqrt( (posx-posx.T)**2 + (posy-posy.T)**2 )
   
    C = np.abs(C)[np.triu_indices_from(C, k=1)]
    PD = PD[np.triu_indices_from(PD, k=1)]

    # plot corr vs dist distribution map
    if nfs is None:
        nfs = np.random.randint(0, C.size, size=9)
        cs = C[nfs].reshape(nfs.size, 1)
        pds = PD[nfs].reshape(nfs.size, 1)
    else:
        cs = np.append(cs, C[nfs].reshape(nfs.size, 1), axis=1)
        pds = np.append(pds, PD[nfs].reshape(nfs.size, 1), axis=1)

print cs
print pds 

nfeat = data.shape[1]
d_mcl = np.linspace(0, 30, 300)
for i in range(len(nfs)):
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1,1,1)
    ax.plot(d_mcl, 1/(d_mcl+1), label="miminum cost line") # minimum cost line
    nf = nfs[i]
    ax.scatter(pds[i,:], np.abs(cs[i,:]))
    for j in range(len(iters)):
        ax.annotate('iter ' + str(iters[j]), (pds[i,j], np.abs(cs[i,j])))
    ax.set_ylabel('Pair-wise feature correlation')
    ax.set_xlabel('Pair-wise physical distance (mm)')
    ax.set_xlim(xmin=0,xmax=30)
    ax.set_ylim(ymin=0,ymax=1)
    ax.set_title('(neuron {0:}, neuron {1:})'.format(nf%nfeat, nf/nfeat))
    plt.savefig('/om/user/hyo/metrics/neural_spatial/' + dir + "/figures/feature_mov" + prefix + '_n' + str(nf) + '.png')
    
