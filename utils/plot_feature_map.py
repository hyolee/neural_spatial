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
iter = sys.argv[3]

net = '/om/user/hyo/metrics/neural_spatial/' + dir + '/train_val_alexnet' + prefix + '.prototxt'
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

# plot feature map
from matplotlib import colors as mcolors
colors = mcolors.cnames.keys()
fig = plt.figure(figsize=(8,8))
for i in range(len(posx.flatten())):
    plt.plot(posx.flatten()[i], posy.flatten()[i], 'o', color=colors[i/64])
ax = plt.gca()
print posx.shape
xmin, xmax = int(np.min(posx)), int(np.max(posx))
ymin, ymax = int(np.min(posy)), int(np.max(posy))
print xmin, xmax
print ymin, ymax
#ax.set_xticks(np.arange(xmin, xmax, 35./posx.shape[0])) #dim_loc[0]))
#ax.set_yticks(np.arange(ymin, ymax, 35./posy.shape[0])) #dim_loc[1]))
ax.set_title('Iteration ' + str(iter))
ax.set_ylim(ymin=min(plt.ylim()[0], -10), ymax=max(plt.ylim()[1], 10))
ax.set_xlim(xmin=min(plt.xlim()[0], -10), xmax=max(plt.xlim()[1], 10))
plt.grid()
plt.savefig('/om/user/hyo/metrics/neural_spatial/' + dir + "/figures/feature_map" + prefix + '_iter' + str(iter) + '.png')

# plot corr vs dist distribution map
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(1,1,1)
im = plot_dist(C, PD, ax, perDist=False, c_nbins=50)
ax.set_title('Iteration ' + str(iter))
plt.savefig('/om/user/hyo/metrics/neural_spatial/' + dir + "/figures/distr_map" + prefix + '_iter' + str(iter) + '.png')

# plot corr vs dist distribution per-dist map
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(1,1,1)
im = plot_dist(C, PD, ax, perDist=True, c_nbins=50)
ax.set_title('Iteration ' + str(iter))
plt.savefig('/om/user/hyo/metrics/neural_spatial/' + dir + "/figures/distr_perdist_map" + prefix + '_iter' + str(iter) + '.png')
# plot loss map
fig = plt.figure(figsize=(8,8))
