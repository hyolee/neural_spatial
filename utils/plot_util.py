import numpy as np
#%matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cPickle
import h5py
import os

def get_map_coord(coord_path, keys, **kwargs):
    # -- coord_path: file path to read/save
    # -- keys: (xkey, ykey), keys in the file coord_path (as a form of "x/y + iteration" i.e. xrandom or x1000)
    # -- method: method of initializing coordinates
    # -- random_seed: random seed for numpy
    # -- feature_shape: shape of the feature matrix
    xkey, ykey = keys
    if os.path.exists(coord_path):
        print "Getting coords from", coord_path, "..."
        f_coord = h5py.File(coord_path, 'r')
        coord_x = f_coord[xkey][:]
        coord_y = f_coord[ykey][:]
    else:
        if len(kwargs) < 3:
            print "Need to give all params!"
            raise
        else:
            print "Computing coords..."
            random_seed = kwargs['random_seed']
            method = kwargs['method']
            feature_shape = kwargs['feature_shape']
            np.random.seed(random_seed)
            # assign as random gaussian, centered at each grid center
            if method=='randomgauss':
                coord_x = np.random.normal(0.5,0.25,list(feature_shape))
                coord_y = np.random.normal(0.5,0.25,list(feature_shape))
                coords = np.array([index for index in np.ndindex(feature_shape)]).reshape(feature_shape+(3,))
                coord_x += coords[:,:,:,1]
                coord_y += coords[:,:,:,2]
                # edge limit
                coord_x = np.abs(coord_x)
                coord_y = np.abs(coord_y)
                coord_x = feature_shape[1] - np.abs(feature_shape[1] - coord_x)
                coord_y = feature_shape[2] - np.abs(feature_shape[2] - coord_y)
            # TODO: assign each feature to the same position in each grid
            elif method=='duplicate':
                print "TODO"
            # save coordinates
            # TODO: check if any duplicate exists
            f_coord = h5py.File(coord_path, 'w')
            f_coord[xkey] = coord_x
            f_coord[ykey] = coord_y
    f_coord.close()
    return coord_x, coord_y

def get_corr(data):
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    Cs = (data - m) / s
    # # get rid of NaNs (TODO: is this right?)
    # self.Cs[:, np.where(self.s==0)] = 0
    C = np.dot(np.transpose(Cs), Cs) / len(data)
    return C

def get_distr(C, PD, perDist=True, d_bsize=1., c_nbins=20.):
    # take upper triangular of absolute correlation matrix and physical distance matrix
    if len(C.shape) > 1:
        c = np.abs(C)[np.triu_indices_from(C, k=1)]
        pd = PD[np.triu_indices_from(C, k=1)]
    else:
        c = C
        pd = PD
    # get rid of nans
    c = c[np.where(~np.isnan(c))]
    pd = pd[np.where(~np.isnan(c))]
    # normalize corr_distr per pd
    dmax=np.max(pd); c_nbins=float(c_nbins)
    ds = np.arange(0,np.max(PD)+d_bsize,d_bsize)
    X = np.zeros((len(ds),c_nbins+1))
    Y = np.zeros((len(ds),c_nbins+1))
    Z = np.zeros((len(ds),c_nbins))
    for i in range(len(ds)):
        d = ds[i]
        s = c[np.where(np.logical_and(d<=pd, pd<d+d_bsize))]
        h = np.histogram(s, bins=c_nbins, range=(0,1))
        X[i,:] = np.array([d]*(len(h[0])+1))         # pd
        Y[i,:] = np.arange(c_nbins+1) / c_nbins         # corr

        if np.sum(h[0])==0:
            Z[i,:] = np.array([0]*len(h[0]))
        elif perDist:
            Z[i,:] = 100.*h[0]/np.sum(h[0])
        else:
            Z[i,:] = 100.*h[0]
            
    return X, Y, Z

def plot_dist(C, pd, ax, perDist=True, d_bsize=1., c_nbins=20.):
    X, Y, Z = get_distr(C, pd, perDist=perDist, d_bsize=d_bsize, c_nbins=c_nbins)
    im = ax.pcolormesh(X, Y, Z)
    plt.colorbar(im, fraction=0.015, pad=0.04)

    ax.set_ylabel('Pair-wise feature correlation')
    ax.set_xlabel('Pair-wise physical distance (mm)')
    ax.set_xlim(xmin=0,xmax=np.max(X))
    ax.set_ylim(ymin=0,ymax=np.max(Y))
    return im

def get_mean_corr(C, PD, ax=None, d_bsize=1.):
    # take upper triangular of absolute correlation matrix and physical distance matrix
    if len(C.shape) > 1:
        c = np.abs(C)[np.triu_indices_from(C, k=1)]
        pd = PD[np.triu_indices_from(C, k=1)]
    else:
        c = C
        pd = PD
    # get rid of nans
    c = c[np.where(~np.isnan(c))]
    pd = pd[np.where(~np.isnan(c))]
    # Get the mean correlation values per distance
    ds = np.arange(0,np.max(pd)+d_bsize,d_bsize)
    s = np.array([])
    d = np.array([])
    for i in range(len(ds)):
        c_here = c[np.where(np.logical_and(ds[i]<=pd, pd<ds[i]+d_bsize))]
        if len(c_here)>0:
    #         print c_here
            d = np.append(d, ds[i])
            s = np.append(s, np.mean(c_here))
    if ax is not None:
        im = ax.plot(d,s)
        ax.set_ylabel('Pair-wise feature correlation')
        ax.set_xlabel('Pair-wise physical distance (mm)')
        ax.set_xlim(xmin=0,xmax=np.max(d))
        ax.set_ylim(ymin=0,ymax=1)
    return (d,s)


# TODO: fix it!
from matplotlib import colors as mcolors
colors = mcolors.cnames.keys()
def plot_feature_map(coord_path, iter, save_path, dim_loc=(10,10), partial=None, nf=None):
    f = h5py.File(coord_path, 'r')
    coord_x = f['x'+str(iter)].value
    coord_y = f['y'+str(iter)].value
    # change PD scale as tissue scale (35mm x 35mm)
    coord_x *= 35. / (coord_x.shape[1] * dim_loc[0])
    coord_y *= 35. / (coord_y.shape[2] * dim_loc[1])

    if partial is not None:
        l, nx, ny = partial
        coord_x = coord_x[:,nx:(nx+l), ny:(ny+l)]
        coord_y = coord_y[:,nx:(nx+l), ny:(ny+l)]
    if nf is not None:
        coord_x = coord_x[:nf, :, :]
        coord_y = coord_y[:nf, :, :]
    else:
        nf = coord_x.shape[0]

    shape = coord_x.shape
    fig = plt.figure(figsize=(8,8))
    for i in range(len(coord_x.flatten())):
        plt.plot(coord_x.flatten()[i], coord_y.flatten()[i], 'o', color=colors[i/nf])
    ax = plt.gca()

    print coord_x.shape
    xmin, xmax = int(np.min(coord_x)), int(np.max(coord_x))
    ymin, ymax = int(np.min(coord_y)), int(np.max(coord_y))
    print xmin, xmax, dim_loc[0]  
    print ymin, ymax, dim_loc[1]
    ax.set_xticks(np.arange(xmin, xmax, 35./coord_x.shape[1])) #dim_loc[0]))
    ax.set_yticks(np.arange(ymin, ymax, 35./coord_y.shape[2])) #dim_loc[1]))
    ax.set_title('Iteration ' + str(iter))
    plt.grid()
    plt.savefig(save_path+"feature_map_" + coord_path.split('coord_')[1].split('.h5')[0] + '_iter' + str(iter) + '.png')

def main():
    # coordinates
    coord_path = sys.argv[1]
    iter = sys.argv[2]
    save_path = sys.argv[3]

if __name__ == "__main__":
    main()
