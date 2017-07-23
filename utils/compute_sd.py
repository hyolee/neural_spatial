import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_util import get_corr, plot_dist, get_mean_corr

feature_path = sys.argv[1]
coord_path = sys.argv[2]
iter = sys.argv[3]
if len(sys.argv) > 4:
    tissue_size = float(sys.argv[4])
else:
    tissue_size = 30.
print "tissue size is", tissue_size, "... is this RIGHT???"

# convert a given array of numbers into coordinates in feature/coord_x/coord_y array, whose shape is (nfil, dim_grid[0], dim_grid[1])
def get_coord(inds, dim_grid):
    return [(ind/np.prod(dim_grid), (ind%np.prod(dim_grid))/dim_grid[1], (ind%np.prod(dim_grid))%dim_grid[1]) for ind in inds]

# assemble feature matrix from given coordinates
def get_resp(feature_path, coords):
    f_feat = h5py.File(feature_path, 'r')
    nfs = np.array(coords)[:,0]
    xs = np.array(coords)[:,1]
    ys = np.array(coords)[:,2]
    bns = ['bn'+str(i) for i in range(23)]
    Fbn = f_feat[bns[0]][:].copy()
    F = Fbn[:,nfs,xs,ys]
    for bn in bns[1:]:
        Fbn = f_feat[bn][:].copy()
        F = np.append(F, Fbn[:,nfs,xs,ys], axis=0)
    if 'hvm' in feature_path:
        F = F[:5760]
    return F

f_coord = h5py.File(coord_path, 'r')
if 'noloc' in coord_path:
    coord_x = f_coord['xrandom'][:].copy()
    coord_y = f_coord['yrandom'][:].copy()
else:
    coord_x = f_coord['x'+iter][:].copy()
    coord_y = f_coord['y'+iter][:].copy()

nfil = coord_x.shape[0]
dim_grid = coord_x.shape[1:]

# get features
if 'rsub' in feature_path:
    f_feat = h5py.File(feature_path, 'r')
    F = f_feat['data'][:].copy()
    ind_coords = f_feat['ind_coords'][:].copy()
    nSample = F.shape[1]
    f_feat.close()
else:
    # randomly sample features
    nSample = 5000
    inds = np.sort(np.random.randint(np.prod(coord_x.shape), size=nSample))
    ind_coords = get_coord(inds, dim_grid)
    print "Getting features from ", feature_path, "..."
    F = get_resp(feature_path, ind_coords)

# get pair-wise correlations
print "Start computing pair-wise correlation..."
C = get_corr(F)
# get pair-wise physical distances
print "Start computing pair-wise physical distances..."
nfs = np.array(ind_coords)[:,0]
xs = np.array(ind_coords)[:,1]
ys = np.array(ind_coords)[:,2]
xc = coord_x[nfs,xs,ys].reshape(nSample,1)
yc = coord_y[nfs,xs,ys].reshape(nSample,1)
PD = np.sqrt( (xc-xc.transpose())**2 + (yc-yc.transpose())**2 )
# change PD scale as tissue scale (35mm x 35mm)
# TODO: currently assuming square!
PD = PD * (tissue_size / dim_grid[0])

# save C, PD
f_feat = h5py.File(feature_path, 'r+')
if 'C' not in f_feat.keys():
    f_feat['C'] = C
else:
    f_feat['C'][:] = C
if 'PD' not in f_feat.keys():
    f_feat['PD'] = PD
else:
    f_feat['PD'][:] = PD
d,s = get_mean_corr(C, PD, d_bsize=2.5)
f_feat['d'] = d
f_feat['s'] = s
f_feat.close()

#print "Plotting..."
## corr distr, normalized per dist
#plt.figure()
#ax = plt.subplot(1,1,1)
#ax.set_aspect('10')
#im = plot_dist(C, PD, ax, d_bsize=2.5)
#plt.title('Randomly sampled ({0:} features)'.format(nSample))
#if save_path is not None:
#    plt.savefig(save_path + 'randomlysampled_' + (feature_path.split('/')[-1]).split('.')[0].split('F_')[-1] + '.png')
## mean corr per dist
#plt.figure()
#ax = plt.subplot(1,1,1)
#ax.set_aspect('10')
#d,s = get_mean_corr(C, PD, ax, d_bsize=2.5)
#plt.title('Randomly sampled ({0:} features)'.format(nSample))
#if save_path is not None:
#    plt.savefig(save_path + 'randomlysampled_' + (feature_path.split('/')[-1]).split('.')[0].split('F_')[-1] + '_meanCorr.png')
#    # save d,s for this iteration
#    if save_iter is not None:
#        from lockfile import LockFile
#        lock = LockFile(save_path + 'randomlysampled_' + (feature_path.split('/')[-1]).split('.')[0].split('F_')[-1].split('_iter')[0] + '_meanCorr.h5')
#        if ('random' in feature_path):
#            iter = 'random'
#        else:
#            iter = (feature_path.split('/')[-1]).split('.')[0].split('F_')[-1].split('_iter')[1]
#        with lock:
#            if os.path.isfile(lock.path):
#                with h5py.File(lock.path, 'r+') as f:
#                    f['d_'+iter] = d
#                    f['s_'+iter] = s
#            else:
#                with h5py.File(lock.path, 'w') as f:
#                    f['d_'+iter] = d
#                    f['s_'+iter] = s
