import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--feature-paths", dest="feature_paths") # example: '/om/user/hyo/metrics/neural_spatial/test_loss3_3/figures/randomlysampled_alexnet_inet_meanCorr.h5'
parser.add_option("-s", "--save-dir", dest="save_dir", default=None) # example: 'alexnet_inet'
parser.add_option("-t", "--labels", dest="labels", default=None)
parser.add_option("-p", "--plot-options", dest="plot_options", default="cdv") # c: correlation distribution, d: distance distribution, v: mean correlation as a function of distance
(options, args) = parser.parse_args()

feature_paths = (options.feature_paths).split()
iters = [feature_path.split('_iter')[1].split('_')[0] for feature_path in feature_paths]
layers = [feature_path.split('_iter')[1].split('_')[1].split('.')[0].split('_')[0] for feature_path in feature_paths]
print [(fp, i, lay) for (fp, i, lay) in zip(feature_paths, iters, layers)]
save_dir = options.save_dir
if save_dir is None:
    save_dir = feature_paths[0].split('/features')[0] + '/figures'
if options.labels is None:
    labels = iters
else:
    labels = (options.labels).split()
plot_options = options.plot_options

fig = {}
ax = {}
for ip in range(len(plot_options)):
    fig[ip] = plt.figure(ip)
    ax[ip] = plt.subplot(1,1,1)

dmax = 0

for i in range(len(feature_paths)):
    feature_path = feature_paths[i]
    iter = iters[i]
    layer = layers[i]
    label = labels[i]

    print "Loading data from ", feature_path, "..."
    f = h5py.File(feature_path, 'r')
    C = f['C'].value
    PD = f['PD'].value
    d = f['d'].value
    s = f['s'].value

    dmax = max(dmax, np.max(PD))
    print PD.shape
    L = np.sum((np.abs(s) - 1)**2 / (2*d)) / len(s)
    print L

    for ip in range(len(plot_options)):
        if 'c' in plot_options[ip]:
            C = C.flatten()
            weights = 100. * np.ones_like(C) / C.size
            im = ax[ip].hist(C, 100, range=(-1,1), histtype='step', label=label, weights=weights)
        elif 'd' in plot_options[ip]:
            PD = PD.flatten()
            weights = 100. * np.ones_like(PD) / PD.size
            im = ax[ip].hist(PD, 100, histtype='step', label=label, weights=weights)
        elif 'v' in plot_options[ip]:
            im = ax[ip].plot(d,s, label=label)
        else: 
            print ip, plot_options[ip], "is a wrong plot option!"
       
 
for ip in range(len(plot_options)):
    if 'c' in plot_options[ip]:
        ax[ip].set_xlabel('Pair-wise feature correlation')
        ax[ip].set_ylabel('% of population')
        ax[ip].set_xlim(xmin=-1,xmax=1)
    if 'd' in plot_options[ip]:
        ax[ip].set_xlabel('Pair-wise physical distance (mm)')
        ax[ip].set_ylabel('% of population')
        ax[ip].set_xlim(xmin=0,xmax=dmax)
    elif 'v' in  plot_options[ip]:
        ax[ip].set_ylabel('Pair-wise feature correlation')
        ax[ip].set_xlabel('Pair-wise physical distance (mm)')
        ax[ip].set_xlim(xmin=0,xmax=dmax)
        ax[ip].set_ylim(ymin=0,ymax=1)
        ax[ip].set_aspect('20')
    else: 
        print ip, plot_options[ip], "is a wrong plot option!"
    ax[ip].legend(bbox_to_anchor=(1.5, 1))
    ax[ip].set_title('Randomly sampled (5000)')
    save_path = save_dir + '/' + plot_options[ip] + '_' + feature_path.split('F_')[1].split('.')[0] + '.png'
    print "Saving plot to", save_path, '...'
    fig[ip].savefig(save_path)
