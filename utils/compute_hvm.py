import cPickle
import h5py
import sys
import numpy as np
import copy

from hvm_analysis import hvm_analysis
import dldata.stimulus_sets.hvm as nd
import dldata.metrics.utils as utils
from hvm_analysis import post_process_neural_regression_msplit
import yamutils.fast as fast

feature_file = sys.argv[1]
nfeat = None
if len(sys.argv) > 2:
    nfeat = int(sys.argv[2])

print "Start computing for " + feature_file

# Get the dataset
hvm_dataset = nd.HvMWithDiscfade()

# Get features
perm = np.random.RandomState(0).permutation(5760)
if '.p' in feature_file:
  features = cPickle.load(open(feature_file))
elif '.h5' in feature_file:
  f_feat = h5py.File(feature_file, 'r')
  bns = ['bn'+str(i) for i in range(23)]
  F = f_feat[bns[0]][:].copy()
  for bn in bns[1:]:
      Fbn = f_feat[bn][:].copy()
      F = np.append(F, Fbn, axis=0)
  if 'hvm' in feature_file:
      F = F[:5760]  
  features = F
else:
  print "Incorrect feature file extension!"

pinv = fast.perminverse(perm)
if len(features.shape) > 2:
  print "reshaping..."
  features = np.reshape(features, (features.shape[0], features.shape[1]*features.shape[2]*features.shape[3]))
if (nfeat is not None) and (features.shape[1] > nfeat):
  print "Select " + str(nfeat) + " features..."
  perm = np.random.RandomState(0).permutation(features.shape[1])
  features = features[:,perm[:nfeat]]
if features.shape[1] > 5760:
  print "random select..."
  perm = np.random.RandomState(0).permutation(features.shape[1])
  features = features[:,perm[:5760]]
features = features[pinv]

## hvm tasks
#print "Computing for hvm tasks..."
#R = hvm_analysis(features, hvm_dataset, ntest=5, do_centroids=False, var_levels=['V6'])
#cPickle.dump( R, open("/om/user/hyo/latent/features/features_matchhuman/results/" + feature_file, "wb") )

# neural fit
print "Computing for neural fits..."
meta = hvm_dataset.meta
IT_feats = hvm_dataset.neuronal_features[:, hvm_dataset.IT_NEURONS]
#V4_feats = hvm_dataset.neuronal_features[:, hvm_dataset.V4_NEURONS]
spec_IT_reg = {'npc_train': 70,
           'npc_test': 10,
           'num_splits': 5,
           'npc_validate': 0,
           'metric_screen': 'regression',
           'metric_labels': None,
           'metric_kwargs': {'model_type': 'pls.PLSRegression',
                             'model_kwargs': {'n_components':25}},
            'labelfunc': lambda x: (IT_feats, None),
            'train_q': {'var':['V3', 'V6']},
            'test_q': {'var':['V3', 'V6']},
            'split_by': 'obj'}
#spec_V4_reg = copy.deepcopy(spec_IT_reg)
#spec_V4_reg['labelfunc'] = lambda  x: (V4_feats, None)

result_IT = utils.compute_metric(features, hvm_dataset, spec_IT_reg)
#result_V4 = utils.compute_metric(features, hvm_dataset, spec_V4_reg)

# noise correction
espec = (('all','','IT_regression'), spec_IT_reg)
post_process_neural_regression_msplit(hvm_dataset, result_IT, espec, n_jobs=1)
#espec = (('all','','V4_regression'), spec_V4_reg)
#post_process_neural_regression_msplit(hvm_dataset, result_V4, espec, n_jobs=1)

print('IT fit', result_IT['noise_corrected_multi_rsquared_loss'])
#print('V4 fit', result_V4['noise_corrected_multi_rsquared_loss'])

#result = {'IT': result_IT, 'V4': result_V4}
output_file = '/'.join(feature_file.split('/')[:-2]) + '/neural_fit/results_' + '_'.join(feature_file.split('/')[-1].split('_')[1:].split('.')[0]) + '.p'
#output_file = "/om/user/hyo/metrics/neural_spatial/dimensionality/features/neural_fit/results_" + '_'.join(feature_file.split('_')[1:])
if nfeat is not None:
    output_file = output_file.split('.')[0] + '_' + str(nfeat) + 'feats.p'
cPickle.dump( result_IT, open(output_file, "wb") )
