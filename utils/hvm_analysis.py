import itertools
import functools
import os
import copy
import math
import numpy as np
import dldata.stimulus_sets.hvm as hvm
from dldata.metrics import utils
from dldata.metrics import classifier
import cPickle
import scipy.stats as stats
import scipy.ndimage.measurements as meas
import pymongo as pm


def hvm_analysis(F, dataset, ntrain=None, ntest=2, num_splits=20, split_by='obj', var_levels=('V3', 'V6'), gridcv=False, attach_predictions=False, attach_models=False, reg_model_kwargs=None, justcat=False, model_type='svm.LinearSVC', model_kwargs=None, do_centroids=True, centroid_sparsity=16, subordinate=True):

    meta = dataset.meta

#    perm = np.random.RandomState(0).permutation(F.shape[1])
#    F = F[:, perm]

    R = {}

    if model_kwargs is None:
        if gridcv:
            model_kwargs = {'GridSearchCV_params': {'C': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4]}}
        else:
            model_kwargs = {'C':5e-3}

    basic_category_eval = {'npc_train': ntrain,
        'npc_test': ntest,
        'num_splits': num_splits,
        'npc_validate': 0,
        'metric_screen': 'classifier',
        'metric_labels': None,
        'metric_kwargs': {'model_type': model_type, 
                          'model_kwargs': model_kwargs
                         },
        'labelfunc': 'category',
        'train_q': {'var': list(var_levels)},
        'test_q': {'var': list(var_levels)},
        'split_by': split_by}


    categories = dataset.categories
    subordinate_evals = {}
    subordinate_evals_sparse = {}
    for c in categories:
        _spec = copy.deepcopy(basic_category_eval)
        _spec['labelfunc'] = 'obj'
        _spec['train_q']['category'] = [c]
        _spec['test_q']['category'] = [c]
        subordinate_evals[c] = _spec

    R['basic_category_result'] = utils.compute_metric(F, dataset, basic_category_eval, attach_models=attach_models, attach_predictions=attach_predictions)
    print('...finished categorization')
    if justcat and not subordinate:
        return R

    if split_by not in ['category']:
        R['subordinate_results'] = dict([(k, utils.compute_metric(F, dataset, v, attach_models=attach_models, attach_predictions=attach_predictions)) for k, v in subordinate_evals.items()])
        print('...finished subordinate')
    else:
        print('... no subordinate')

    if justcat:
        return R

    masks = dataset.pixel_masks

    centroids = np.array(map(meas.center_of_mass, masks))
    centroids[np.isnan(centroids[:, 0])] = -128

    volumes = masks.sum(1).sum(1)

    def get_axis_bb(x):
        nzx, nzy = x.nonzero()
        if len(nzx):
            return (nzy.min(), nzx.min(), nzy.max(), nzx.max())
        else:
            return (-128, -128, -128, -128)

    axis_bb = np.array(map(get_axis_bb, masks))

    get_axis_bb_ctr = lambda x: ((x[1] + x[3])/2, (x[0] + x[2])/2)

    axis_bb_ctr = np.array(map(get_axis_bb_ctr, axis_bb))

    axis_bb_sx = np.abs(axis_bb[:, 2] - axis_bb[:, 0])
    axis_bb_sy = np.abs(axis_bb[:, 3] - axis_bb[:, 1])
    axis_bb_area = axis_bb_sx * axis_bb_sy

    axis_bb_sx = np.abs(axis_bb[:, 2] - axis_bb[:, 0])
    axis_bb_sy = np.abs(axis_bb[:, 3] - axis_bb[:, 1])
    axis_bb_area = axis_bb_sx * axis_bb_sy

    borders = dataset.pixel_borders(thickness=1)
    area_bb = np.array(map(hvm.get_best_box, borders))
    area_bb[3818] = -128

    def line(a1, a2):
        if a1[0] != a2[0] and a1[1] != a2[1]:
            m = (a2[1] - a1[1]) / float(a2[0] - a1[0])
            b = a1[1] - m * a1[0]
        elif a1[1] == a2[1]:
            m = np.inf
            b = np.nan
        elif a1[0] == a2[0]:
            m = 0
            b = a1[0]
        return m, b

    def intersection(a):
        a1, a2, a3, a4 = a
        m1, b1 = line(a1, a4)
        m2, b2 = line(a2, a3)
        if not np.isinf(m1):
            if not np.isinf(m2):
                ix = (b2 - b1)/(m1 - m2)
            else:
                ix = a3[1]
            iy = m1 * ix + b1
        else:
            ix = a1[1]
            iy = m2 * ix + b2
        return ix, iy


    area_bb_ctr = np.array(map(intersection, area_bb))
    dist = lambda _b1, _b2: math.sqrt((_b1[0] - _b2[0])**2 + (_b1[1] - _b2[1])**2)
    def sminmax(a):
        a1, a2, a3, a4 = a
        s1 = dist(a1, a2)
        s2 = dist(a1, a3)
        smin = min(s1, s2)
        smax = max(s1, s2)
        return smin, smax

    area_bb_minmax = np.array(map(sminmax, area_bb))
    area_bb_area = area_bb_minmax[:, 0] * area_bb_minmax[:, 1]

    dist = lambda _b1, _b2: np.sqrt((_b1[0] - _b2[0])**2 + (_b1[1] - _b2[1])**2)

    def get_major_axis(a):
        a1, a2, a3, a4 = a
        s1 = dist(a1, a2)
        s2 = dist(a1, a3)
        if s1 > s2:
            m11 = a1
            m12 = a3
            m21 = a2
            m22 = a4
        else:
            m11 = a1
            m12 = a2
            m21 = a3
            m22 = a4

        mj1 = (m11 + m12)/2
        mj2 = (m21 + m22)/2
        return (mj1, mj2)

    def nonan(f):
        f[np.isnan(f)] = 0
        return f

    area_bb_major_axes = np.array(map(get_major_axis, area_bb))
    sin_maj = nonan((area_bb_major_axes[:, 0, 1] - area_bb_major_axes[:, 1, 1]) / dist(area_bb_major_axes[:, 0].T, area_bb_major_axes[:, 1].T))
    cos_maj = nonan((area_bb_major_axes[:, 0, 0] - area_bb_major_axes[:, 1, 0]) / dist(area_bb_major_axes[:, 0].T, area_bb_major_axes[:, 1].T))

    if reg_model_kwargs is None:
        reg_model_kwargs = {'alphas': [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, .75e-1, 1e0, 2.5e0, 5e0, 1e1, 25, 1e2, 1e3]}

    def make_spec(f):
        x = {'npc_train': ntrain,
        'npc_test': ntest,
        'num_splits': num_splits,
        'npc_validate': 0,
        'metric_screen': 'regression',
        'metric_labels': None,
        'metric_kwargs': {'model_type': 'linear_model.RidgeCV',
                          'model_kwargs': reg_model_kwargs},
        'train_q': {'var': list(var_levels)},
        'test_q': {'var': list(var_levels)},
        'split_by': split_by}

        x['labelfunc'] = f
        return x

    
    if do_centroids:
        distctr = lambda m, x : (np.sqrt((centroids[:, 0] - m[0])**2 + (centroids[:, 1] - m[1])**2), None)
        centroid_specs =  dict([((i, j), make_spec(functools.partial(distctr, (i, j)))) for i in np.arange(256)[:: centroid_sparsity] for j in np.arange(256)[:: centroid_sparsity]])
        R['centroid_results'] = dict([(k, utils.compute_metric(F, dataset, centroid_specs[k], attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)) for k in centroid_specs])
        print('... finished centroid')
    else:
        print('... no centroids')

    volume_spec = make_spec(lambda x: (volumes, None))
    R['volume_result'] = utils.compute_metric(F, dataset, volume_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished volume')

    perimeters = borders.sum(1).sum(1)
    perimeter_spec = make_spec(lambda x: (perimeters, None))
    R['perimeter_result'] = utils.compute_metric(F, dataset, perimeter_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished perimeter')

    axis_bb_sx_spec = make_spec(lambda x: (axis_bb_sx, None))
    axis_bb_sy_spec = make_spec(lambda x: (axis_bb_sy, None))
    axis_bb_asp_spec = make_spec(lambda x: (axis_bb_sx / np.maximum(axis_bb_sy, 1e-5), None))
    axis_bb_area_spec = make_spec(lambda x: (axis_bb_area, None))

    R['axis_bb_sx_result'] = utils.compute_metric(F, dataset, axis_bb_sx_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished axis_bb_sx')
    R['axis_bb_sy_result'] = utils.compute_metric(F, dataset, axis_bb_sy_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished axis_bb_sy')
    R['axis_bb_asp_result'] = utils.compute_metric(F, dataset, axis_bb_asp_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished axis_bb_asp')
    R['axis_bb_area_result'] = utils.compute_metric(F, dataset, axis_bb_area_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... finished axis_bb_area')

    area_bb_smin_spec = make_spec(lambda x: (area_bb_minmax[:, 0], None))
    area_bb_smax_spec = make_spec(lambda x: (area_bb_minmax[:, 1], None))

    area_bb_asps = area_bb_minmax[:, 1]/np.maximum(area_bb_minmax[:, 0], 1e-5)
    area_bb_asp_spec = make_spec(lambda x: (area_bb_asps, None))
    area_bb_area_spec = make_spec(lambda x: (area_bb_area, None))

    sinmajs = np.abs(sin_maj)
    sinmaj_spec = make_spec(lambda x: (sinmajs, None))

    R['area_bb_smin_result'] = utils.compute_metric(F, dataset, area_bb_smin_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... area_bb_smin')
    R['area_bb_smax_result'] = utils.compute_metric(F, dataset, area_bb_smax_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... area_bb_smax')
    R['area_bb_asp_result'] = utils.compute_metric(F, dataset, area_bb_asp_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... area_bb_asp')
    R['area_bb_area_result'] = utils.compute_metric(F, dataset, area_bb_area_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... area_bb_area')
    R['sinmaj_result'] = utils.compute_metric(F, dataset, sinmaj_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... sinmaj')

    #position
    posx_spec = make_spec(lambda x: (x['ty'], None))
    posy_spec = make_spec(lambda x: (x['tz'], None))
    R['posx_result'] = utils.compute_metric(F, dataset, posx_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    R['posy_result'] = utils.compute_metric(F, dataset, posy_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)


    size_spec = make_spec(lambda x: (x ['s'], None))
    R['size_result'] = utils.compute_metric(F, dataset, size_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... size')

    masksv0 = masks[meta['var'] == 'V0'][::10]
    objs = meta['obj'][meta['var'] == 'V0'][::10]
    A = np.array([_x.nonzero()[0][[0, -1]] if _x.sum() > 0 else (0, 0) for _x in masksv0.sum(1) ])
    B = np.array([_x.nonzero()[0][[0, -1]] if _x.sum() > 0 else (0, 0) for _x in masksv0.sum(2) ])
    base_size_factors = dict(zip( objs, np.maximum(A[:, 1] - A[:, 0], B[:, 1] - B[:, 0])))
    size_factors = np.zeros(len(meta))
    for o in objs:
        inds = [meta['obj'] == o]
        size_factors[inds] = base_size_factors[o] * .01
    scaled_size  = size_factors * meta['s']
    scaled_size_spec = make_spec(lambda x: (scaled_size, None))
    R['scaled_size_result'] = utils.compute_metric(F, dataset, scaled_size_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... scaled size')

    rxy_spec = make_spec(lambda x: (x['rxy_semantic'], None))
    rxz_spec = make_spec(lambda x: (x['rxz_semantic'], None))
    ryz_spec = make_spec(lambda x: (x['ryz_semantic'], None))
    R['rxy_result'] = utils.compute_metric(F, dataset, rxy_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    R['rxz_result'] = utils.compute_metric(F, dataset, rxz_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    R['ryz_result'] = utils.compute_metric(F, dataset, ryz_spec, attach_models=attach_models, attach_predictions=attach_predictions, return_splits=False)
    print('... rotations')

    subordinate_rxy_specs = {}
    for c in categories:
        _spec = copy.deepcopy(rxy_spec)
        _spec['train_q']['category'] = [c]
        _spec['test_q']['category'] = [c]
        subordinate_rxy_specs[c] = _spec

    subordinate_rxz_specs = {}
    for c in categories:
        _spec = copy.deepcopy(rxz_spec)
        _spec['train_q']['category'] = [c]
        _spec['test_q']['category'] = [c]
        subordinate_rxz_specs[c] = _spec

    subordinate_ryz_specs = {}
    for c in categories:
        _spec = copy.deepcopy(ryz_spec)
        _spec['train_q']['category'] = [c]
        _spec['test_q']['category'] = [c]
        subordinate_ryz_specs[c] = _spec

    if split_by not in ['category']:
        R['subordinate_rxy_results'] = dict([(k, utils.compute_metric(F, dataset, v, attach_models=attach_models, attach_predictions=attach_predictions)) for k, v in subordinate_rxy_specs.items()])
        R['subordinate_rxz_results'] = dict([(k, utils.compute_metric(F, dataset, v, attach_models=attach_models, attach_predictions=attach_predictions)) for k, v in subordinate_rxz_specs.items()])
        R['subordinate_ryz_results'] = dict([(k, utils.compute_metric(F, dataset, v, attach_models=attach_models, attach_predictions=attach_predictions)) for k, v in subordinate_ryz_specs.items()])
        print('... subordinate rotations')

    return R

hvm_dataset = hvm.HvMWithDiscfade()
IT_feats = hvm_dataset.neuronal_features[:, hvm_dataset.IT_NEURONS]
V4_feats = hvm_dataset.neuronal_features[:, hvm_dataset.V4_NEURONS]

reg_eval_it =  {'labelfunc': lambda x: (IT_feats, None),
  'metric_kwargs': {'model_kwargs': {'n_components': 25},
  'model_type': 'pls.PLSRegression'},
  'metric_labels': None,
  'metric_screen': 'regression',
  'npc_test': 10,
  'npc_train': 70,
  'npc_validate': 0,
  'num_splits': 5,
  'split_by': 'obj',
  'test_q': {'var': ['V3', 'V6']},
  'train_q': {'var': ['V3', 'V6']}}

reg_eval_IT_obj_half = {'labelfunc': lambda x: (IT_feats, None),
  'metric_kwargs': {'model_kwargs': {'n_components': 25},
  'model_type': 'pls.PLSRegression'},
  'metric_labels': None,
  'metric_screen': 'regression',
  'npc_test': 75,
  'npc_train': 80,
  'npc_validate': 0,
  'num_splits': 5,
  'split_by': 'obj',
  'test_q': {'obj': hvm.OBJECT_SET_2, 'var': ['V3', 'V6']},
  'train_q': {'obj': hvm.OBJECT_SET_1}, 'var': ['V3', 'V6']}
  
  
reg_eval_IT_category_half = {'labelfunc': lambda x: (IT_feats, None),
  'metric_kwargs': {'model_kwargs': {'n_components': 25},
  'model_type': 'pls.PLSRegression'},
  'metric_labels': None,
  'metric_screen': 'regression',
  'npc_test': 75,
  'npc_train': 80,
  'npc_validate': 0,
  'num_splits': 5,
  'split_by': 'obj',
  'train_q': {'category': hvm_dataset.categories[::2], 'var': ['V3', 'V6']},
  'test_q': {'category': hvm_dataset.categories[1::2]}, 'var': ['V3', 'V6']}
  
reg_eval_v4 =  {'labelfunc': lambda x: (V4_feats, None),
  'metric_kwargs': {'model_kwargs': {'n_components': 25},
  'model_type': 'pls.PLSRegression'},
  'metric_labels': None,
  'metric_screen': 'regression',
  'npc_test': 10,
  'npc_train': 70,
  'npc_validate': 0,
  'num_splits': 5,
  'split_by': 'obj',
  'test_q': {'var': ['V3', 'V6']},
  'train_q': {'var': ['V3', 'V6']}}


def post_process_neural_regression_msplit(dataset, result, spec, n_jobs=1, splits=None):
    name = spec[0]
    specval = spec[1]
    assert name[2] in ['IT_regression', 'V4_regression', 'ITc_regression', 'ITt_regression'], name
    if name[2] == 'IT_regression':
        units = dataset.IT_NEURONS
    elif name[2] == 'ITc_regression':
        units = hvm.mappings.LST_IT_Chabo
    elif name[2] == 'ITt_regression':
        units = hvm.mappings.LST_IT_Tito
    else:
        units = dataset.V4_NEURONS

    units = np.array(units)
    if not splits:
        splits, validations = utils.get_splits_from_eval_config(specval, dataset)

    sarrays = []
    for s_ind, s in enumerate(splits):
        ne = dataset.noise_estimate(s['test'], units=units, n_jobs=n_jobs, cache=True)
        farray = np.asarray(result['split_results'][s_ind]['test_multi_rsquared'])
        sarray = farray / ne[0]**2
        sarrays.append(sarray)
    sarrays = np.asarray(sarrays)
    result['noise_corrected_multi_rsquared_array_loss'] = (1 - sarrays).mean(0)
    result['noise_corrected_multi_rsquared_array_loss_stderror'] = (1 - sarrays).std(0)
    result['noise_corrected_multi_rsquared_loss'] = np.median(1 - sarrays, 1).mean()
    result['noise_corrected_multi_rsquared_loss_stderror'] = np.median(1 - sarrays, 1).std()


def post_process_neural_regression_msplit_aggregate(dataset, results, spec, n_jobs=1, splits=None):
    name = spec[0]
    specval = spec[1]
    assert name[2] in ['IT_regression', 'V4_regression', 'ITc_regression', 'ITt_regression'], name
    if name[2] == 'IT_regression':
        units = dataset.IT_NEURONS
    elif name[2] == 'ITc_regression':
        units = hvm.mappings.LST_IT_Chabo
    elif name[2] == 'ITt_regression':
        units = hvm.mappings.LST_IT_Tito
    else:
        units = dataset.V4_NEURONS

    units = np.array(units)
    if not splits:
        splits, validations = utils.get_splits_from_eval_config(specval, dataset)

    aggregate_result = {}
    mrs_array = np.array([[r['split_results'][sind]['test_rsquared'] for r in results] for sind in range(len(splits))])
    mrs_array_loss = (1 - mrs_array).mean(0)
    mrs_array_loss_stderror = (1 - mrs_array).std(0)
    mrs_loss = (1 - np.median(mrs_array, 1)).mean(0)
    mrs_loss_stderror = (1 - np.median(mrs_array, 1)).std(0)
    aggregate_result['multi_rsquared_array_loss'] = mrs_array_loss
    aggregate_result['multi_rsquared_array_loss_stderror'] = mrs_array_loss_stderror
    aggregate_result['multi_rsquared_loss'] = mrs_loss
    aggregate_result['multi_rsquared_loss_stderror'] = mrs_loss_stderror

    aggregate_result['corr_loss'] = np.mean([1 - np.median([r['split_results'][sind]['test_corr'] for r in results]) for sind in range(len(splits))])
    aggregate_result['corr_loss_stderror'] = np.std([1 - np.median([r['split_results'][sind]['test_corr'] for r in results]) for sind in range(len(splits))])
    
    sarrays = []
    for s_ind, s in enumerate(splits):
        ne = dataset.noise_estimate(s['test'], units=units, n_jobs=n_jobs, cache=True)
        #assert len(ne[0]) == len(results), (len(ne[0]), len(results))
        print("LE", len(ne[0]), len(results))
        nvec = ne[0][:len(results)]
        farray = np.asarray([r['split_results'][s_ind]['test_rsquared'] for r in results])
        sarray = farray / nvec**2
        sarrays.append(sarray)
    sarrays = np.asarray(sarrays)

    aggregate_result['noise_corrected_multi_rsquared_array_loss'] = (1 - sarrays).mean(0)
    aggregate_result['noise_corrected_multi_rsquared_array_loss_stderror'] = (1 - sarrays).std(0)
    aggregate_result['noise_corrected_multi_rsquared_loss'] = np.median(1 - sarrays, 1).mean()
    aggregate_result['noise_corrected_multi_rsquared_loss_stderror'] = np.median(1 - sarrays, 1).std()

    return aggregate_result


def get_neural_fit_lasso(F):
    
    reg_evals_it =  [{'labelfunc': functools.partial(lambda _i, x: (IT_feats[:, _i], None), i),
                    'metric_kwargs': {'model_kwargs': {'alphas': [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, .75e-1, 1e0, 2.5e0, 5e0, 1e1, 25, 1e2, 1e3]},
                                      'model_type': 'linear_model.LassoCV'},
                    'metric_labels': None,
                    'metric_screen': 'regression',
                    'npc_test': 10,
                    'npc_train': 70,
                    'npc_validate': 0,
                    'num_splits': 5,
                    'split_by': 'obj',
                    'test_q': {'var': ['V3', 'V6']},
                    'train_q': {'var': ['V3', 'V6']}} for i in range(IT_feats.shape[1])]
    
    reg_evals_IT_obj_half = [{'labelfunc': functools.partial(lambda _i, x: (IT_feats[:, _i], None), i),
                            'metric_kwargs': {'model_kwargs': {'alphas': [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, .75e-1, 1e0, 2.5e0, 5e0, 1e1, 25, 1e2, 1e3]},
                                              'model_type': 'linear_model.LassoCV'},
                            'metric_labels': None,
                            'metric_screen': 'regression',
                            'npc_test': 75,
                            'npc_train': 80,
                            'npc_validate': 0,
                            'num_splits': 5,
                            'split_by': 'obj',
                            'test_q': {'obj': hvm.OBJECT_SET_2, 'var': ['V3', 'V6']},
                            'train_q': {'obj': hvm.OBJECT_SET_1}, 'var': ['V3', 'V6']} for i in range(IT_feats.shape[1])]
  
  
    reg_evals_IT_category_half = [{'labelfunc': functools.partial(lambda _i, x: (IT_feats[:, _i], None), i),
                                 'metric_kwargs': {'model_kwargs': {'alphas': [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, .75e-1, 1e0, 2.5e0, 5e0, 1e1, 25, 1e2, 1e3]},
                                                  'model_type': 'linear_model.LassoCV'},
                                 'metric_labels': None,
                                 'metric_screen': 'regression',
                                 'npc_test': 75,
                                 'npc_train': 80,
                                 'npc_validate': 0,
                                 'num_splits': 5,
                                 'split_by': 'obj',
                                 'train_q': {'category': hvm_dataset.categories[::2], 'var': ['V3', 'V6']},
                                 'test_q': {'category': hvm_dataset.categories[1::2]}, 'var': ['V3', 'V6']} for i in range(IT_feats.shape[1])]
  
    reg_evals_v4 =  [{'labelfunc': functools.partial(lambda _i, x: (V4_feats[:, _i], None), i),
                    'metric_kwargs': {'model_kwargs': {'alphas': [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, .75e-1, 1e0, 2.5e0, 5e0, 1e1, 25, 1e2, 1e3]},
                                      'model_type': 'linear_model.LassoCV'},
                    'metric_labels': None,
                    'metric_screen': 'regression',
                    'npc_test': 10,
                    'npc_train': 70,
                    'npc_validate': 0,
                    'num_splits': 5,
                    'split_by': 'obj',
                    'test_q': {'var': ['V3', 'V6']},
                    'train_q': {'var': ['V3', 'V6']}} for i in range(V4_feats.shape[1])]


    resits = [utils.compute_metric(F, hvm_dataset, e, attach_predictions=False) for e in reg_evals_it]
    espec = (('all','','IT_regression'), e)
    resit = post_process_neural_regression_msplit_aggregate(hvm_dataset, resits, espec, n_jobs=1)
    print('IT fit', 1-resit['noise_corrected_multi_rsquared_loss'])


    resits_o = [utils.compute_metric(F, hvm_dataset, e, attach_predictions=False) for e in reg_evals_IT_obj_half]
    espec = (('all','','IT_regression'), e)
    resit_o = post_process_neural_regression_msplit_aggregate(hvm_dataset, resits_o, espec, n_jobs=1)

    resits_c = [utils.compute_metric(F, hvm_dataset, e, attach_predictions=False) for e in reg_evals_IT_category_half]
    espec = (('all','','IT_regression'), e)
    resit_c = post_process_neural_regression_msplit_aggregate(hvm_dataset, resits_c, espec, n_jobs=1)

    resv4s = [utils.compute_metric(F, hvm_dataset, e, attach_predictions=False) for e in reg_evals_v4]
    espec = (('all','','V4_regression'), e)
    resv4 = post_process_neural_regression_msplit_aggregate(hvm_dataset, resv4s, espec, n_jobs=1)
    print('V4 fit', 1-resv4['noise_corrected_multi_rsquared_loss'])
    return {'IT': resit, 'V4': resv4, 'IT_o': resit_o, 'IT_c': resit_c} 


def get_neural_fit(F, n_components=None):
    e = reg_eval_it
    if n_components is not None:
        e = copy.deepcopy(e)
        e['metric_kwargs']['model_kwargs']['n_components'] = n_components
    resit = utils.compute_metric(F, hvm_dataset, e, attach_predictions=False)
    espec = (('all','','IT_regression'), e)
    post_process_neural_regression_msplit(hvm_dataset, resit, espec, n_jobs=1)
    print('IT fit', 1-resit['noise_corrected_multi_rsquared_loss'])

    e = reg_eval_IT_obj_half
    if n_components is not None:
        e = copy.deepcopy(e)
        e['metric_kwargs']['model_kwargs']['n_components'] = n_components
    resit_o = utils.compute_metric(F, hvm_dataset, e, attach_predictions=False)
    espec = (('all','','IT_regression'), e)
    post_process_neural_regression_msplit(hvm_dataset, resit_o, espec, n_jobs=1)
    print('ito fit', 1-resit_o['noise_corrected_multi_rsquared_loss'])

    e = reg_eval_IT_category_half
    if n_components is not None:
        e = copy.deepcopy(e)
        e['metric_kwargs']['model_kwargs']['n_components'] = n_components
    resit_c = utils.compute_metric(F, hvm_dataset, e, attach_predictions=False)
    espec = (('all','','IT_regression'), e)
    post_process_neural_regression_msplit(hvm_dataset, resit_c, espec, n_jobs=1)

    e = reg_eval_v4
    if n_components is not None:
        e = copy.deepcopy(e)
        e['metric_kwargs']['model_kwargs']['n_components'] = n_components
    resv4 = utils.compute_metric(F, hvm_dataset, e, attach_predictions=False)
    espec = (('all','','V4_regression'), e)
    post_process_neural_regression_msplit(hvm_dataset, resv4, espec, n_jobs=1)
    print('V4 fit', 1-resv4['noise_corrected_multi_rsquared_loss'])

    return {'IT': resit, 'V4': resv4, 'IT_o': resit_o, 'IT_c': resit_c}


def pr(X):
    return [x[:, :, np.newaxis] for x in X]


def consistency_from_cms(*args):
    return utils.consistency_from_confusion_mat(*args, error=None)   


def get_basic_analysis_with_models(F):
    basic_res = hvm_analysis(F, hvm_dataset, ntrain=None, ntest=5, num_splits=5, split_by='obj', var_levels=('V3', 'V6'), gridcv=False, attach_predictions=False, attach_models=True, reg_model_kwargs=None, justcat=False, model_type='svm.LinearSVC', model_kwargs=None, do_centroids=True, centroid_sparsity=64)
    return basic_res


def hvm_analysis_api(F, **kwargs):
    return hvm_analysis(F, hvm_dataset, **kwargs)


def get_fit_and_basic(F):
    R = {}
    fit_res = get_neural_fit(F)
    R['fit_res'] = fit_res
    R['performance'] =  hvm_analysis(F, hvm_dataset, ntrain=None, ntest=5, num_splits=5, split_by='obj', var_levels=('V3', 'V6'), gridcv=False, attach_predictions=False, attach_models=False, reg_model_kwargs=None, justcat=True, model_type='svm.LinearSVC', model_kwargs=None, subordinate=False)
    return R


def get_all_analysis(F):

    R = {}

    fit_res = get_neural_fit(F)
    R['fit_res'] = fit_res

    basic_res = hvm_analysis(F, hvm_dataset, ntrain=None, ntest=5, num_splits=5, split_by='obj', var_levels=('V3', 'V6'), gridcv=False, attach_predictions=False, attach_models=False, reg_model_kwargs=None, justcat=False, model_type='svm.LinearSVC', model_kwargs=None, do_centroids=False)
    R['basic_res'] = basic_res

    cat_v0_res = hvm_analysis(F, hvm_dataset, ntrain=None, ntest=5, num_splits=5, split_by='obj', var_levels=('V0', ), gridcv=False, attach_predictions=False, attach_models=False, reg_model_kwargs=None, justcat=True, model_type='svm.LinearSVC', model_kwargs=None, do_centroids=False)
    R['v0_res'] = cat_v0_res

    cat_v3_res = hvm_analysis(F, hvm_dataset, ntrain=None, ntest=20, num_splits=5, split_by='obj', var_levels=('V3', ), gridcv=False, attach_predictions=False, attach_models=False, reg_model_kwargs=None, justcat=True, model_type='svm.LinearSVC', model_kwargs=None, do_centroids=False)
    R['v3_res'] = cat_v3_res

    cat_v6_res = hvm_analysis(F, hvm_dataset, ntrain=None, ntest=20, num_splits=5, split_by='obj', var_levels=('V6', ), gridcv=False, attach_predictions=False, attach_models=False, reg_model_kwargs=None, justcat=True, model_type='svm.LinearSVC', model_kwargs=None, do_centroids=False)
    R['v6_res'] = cat_v6_res

    model_cms = [cat_v0_res['basic_category_result']['result_summary']['cms'],
                 cat_v3_res['basic_category_result']['result_summary']['cms'],
                 cat_v6_res['basic_category_result']['result_summary']['cms']] + \
                list(itertools.chain(*[[cat_v0_res['subordinate_results'][k]['result_summary']['cms'],
                                        cat_v3_res['subordinate_results'][k]['result_summary']['cms'],
                                        cat_v6_res['subordinate_results'][k]['result_summary']['cms']] for k in hvm_dataset.categories]))
    model_cms_standard = model_cms[:3] + model_cms[3 * 3: 3 * 4] + model_cms[3 * 5: 3 * 6]
    model_cms_basic = model_cms[:3]
    model_cms_v0 = model_cms[::3]
    model_cms_v6 = model_cms[2::3]

    conn = pm.Connection(port=22334)

    colname = 'hvm_basic_categorization_new'
    query = {}
    threshold = 0
    tr_period = 10
    get_pred = lambda x: x[0]['dataChosen']['category']
    get_actual = lambda x: x[1]['Sample']['category']

    idataquery = lambda x: x[1]['Sample']['var'] == 'V0'
    labelset = cat_v0_res['basic_category_result']['result_summary']['labelset']
    human_cm_0 = get_human_cm(colname, query, tr_period, get_pred, get_actual, idataquery, conn, labelset=labelset)

    idataquery = lambda x: x[1]['Sample']['var'] == 'V3'
    labelset = cat_v3_res['basic_category_result']['result_summary']['labelset']
    human_cm_3 = get_human_cm(colname, query, tr_period, get_pred, get_actual, idataquery, conn, labelset=labelset)

    idataquery = lambda x: x[1]['Sample']['var'] == 'V6'
    labelset = cat_v6_res['basic_category_result']['result_summary']['labelset']
    human_cm_6 = get_human_cm(colname, query, tr_period, get_pred, get_actual, idataquery, conn, labelset=labelset)

    human_cms = [human_cm_0, human_cm_3, human_cm_6]

    for cat in hvm_dataset.categories:
        colname = 'hvm_subordinate_identification_%s' % cat
        query = {}
        threshold = 0
        tr_period = 10
        get_pred = lambda x: x[0]['dataChosen']['obj']
        get_actual = lambda x: x[1]['Sample']['obj']
    
        idataquery = lambda x: x[1]['Sample']['var'] == 'V0'
        labelset = cat_v0_res['subordinate_results'][cat]['result_summary']['labelset']
        m0 = get_human_cm(colname, query, tr_period, get_pred, get_actual, idataquery, conn, labelset)

        idataquery = lambda x: x[1]['Sample']['var'] == 'V3'
        labelset = cat_v3_res['subordinate_results'][cat]['result_summary']['labelset']
        m3 = get_human_cm(colname, query, tr_period, get_pred, get_actual, idataquery, conn, labelset)
     
        idataquery = lambda x: x[1]['Sample']['var'] == 'V6'
        labelset = cat_v6_res['subordinate_results'][cat]['result_summary']['labelset']
        m6 = get_human_cm(colname, query, tr_period, get_pred, get_actual, idataquery, conn, labelset)
    
        human_cms.extend([m0, m3, m6])

    human_cms_standard = human_cms[:3] + human_cms[3 * 3: 3 * 4] + human_cms[3 * 5: 3 * 6]
    human_cms_basic = human_cms[:3]
    human_cms_v0 = human_cms[::3]
    human_cms_v6 = human_cms[2::3]

    C_standard = consistency_from_cms(model_cms_standard, pr(human_cms_standard))
    C_basic = consistency_from_cms(model_cms_basic, pr(human_cms_basic))
    C_all = consistency_from_cms(model_cms, pr(human_cms))
    C_v0 = consistency_from_cms(model_cms_v0, pr(human_cms_v0))
    C_v6 = consistency_from_cms(model_cms_v6, pr(human_cms_v6))

    consistency = {"consistency_standard": C_standard, "consistency_basic": C_basic, "consistency_all": C_all, "consistency_v0": C_v0, "consistency_v6": C_v6}
    R.update(consistency)


    face_dprimes_all = get_face_dprimes(F, var_level=None)
    face_dprimes_v0 = get_face_dprimes(F, var_level='V0')
    face_dprimes_v3 = get_face_dprimes(F, var_level='V3')
    face_dprimes_v6 = get_face_dprimes(F, var_level='V6')

    e = copy.deepcopy(reg_eval_IT_obj_half)
    e['num_splits'] = 1
    e['test_q'].pop('var')
    e['npc_test'] = 90
    resit_o = utils.compute_metric(F, hvm_dataset, e, attach_predictions=True, return_splits=True)
    pop = np.array(resit_o['split_results'][0]['test_prediction'])
    ts = np.array(resit_o['splits'][0][0]['test'])
    dpv0 = get_face_dprimes(pop, var_level='V0', inds=ts)
    dpv3 = get_face_dprimes(pop, var_level='V3', inds=ts)
    dpv6 = get_face_dprimes(pop, var_level='V6', inds=ts)
    dpall = get_face_dprimes(pop, var_level=None, inds=ts)


    thresholds = [.25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 5]
    face_selectivity_levels = {}
    def count(x, thr):
        n = len((x > thr).nonzero()[0]) * 100
        return float(n) / len(x)
    for thr in thresholds:
        face_selectivity_levels[(None, thr)] = count(face_dprimes_all, thr)
        face_selectivity_levels[('V0', thr)] = count(face_dprimes_v0, thr)
        face_selectivity_levels[('V3', thr)] = count(face_dprimes_v3, thr)
        face_selectivity_levels[('V6', thr)] = count(face_dprimes_v6, thr)
        face_selectivity_levels[('fitpop', None, thr)] = count(dpall, thr)
        face_selectivity_levels[('fitpop', 'V0', thr)] = count(dpv0, thr)
        face_selectivity_levels[('fitpop', 'V3', thr)] = count(dpv3, thr)
        face_selectivity_levels[('fitpop', 'V6', thr)] = count(dpv6, thr)

    face_stats = {'face_dprimes_all': face_dprimes_all, 
                  'face_dprimes_v0': face_dprimes_v0,
                  'face_dprimes_v3': face_dprimes_v3,
                  'face_dprimes_v6': face_dprimes_v6,
                  'fitpop_face_dprimes_all': dpall,
                  'fitpop_face_dprimes_v0': dpv0, 
                  'fitpop_face_dprimes_v3': dpv3,
                  'fitpop_face_dprimes_v6': dpv6,
                  'selectivities': face_selectivity_levels}

    R['face_statistics'] = face_stats
    return R


def get_fit_analysis(F):

    R = {}

    for n in [2, 5, 10, 15, 20, 25, 30]:
        fit_res = get_neural_fit(F, n_components = n)
        R[('fit_res', n)] = fit_res

        e = copy.deepcopy(reg_eval_IT_obj_half)
        e['metric_kwargs']['model_kwargs']['n_components'] = n
        e['num_splits'] = 1
        e['test_q'].pop('var')
        e['npc_test'] = 90
        resit_o = utils.compute_metric(F, hvm_dataset, e, attach_predictions=True, return_splits=True)
        pop = np.array(resit_o['split_results'][0]['test_prediction'])
        ts = np.array(resit_o['splits'][0][0]['test'])
        dpv0 = get_face_dprimes(pop, var_level='V0', inds=ts)
        dpv3 = get_face_dprimes(pop, var_level='V3', inds=ts)
        dpv6 = get_face_dprimes(pop, var_level='V6', inds=ts)
        dpall = get_face_dprimes(pop, var_level=None, inds=ts)

        thresholds = [.25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 5]
        face_selectivity_levels = {}
        def count(x, thr):
            n = len((x > thr).nonzero()[0]) * 100
            return float(n) / len(x)
        for thr in thresholds:
            face_selectivity_levels[('fitpop', None, thr)] = count(dpall, thr)
            face_selectivity_levels[('fitpop', 'V0', thr)] = count(dpv0, thr)
            face_selectivity_levels[('fitpop', 'V3', thr)] = count(dpv3, thr)
            face_selectivity_levels[('fitpop', 'V6', thr)] = count(dpv6, thr)

        face_stats = {
                  'fitpop_face_dprimes_all': dpall,
                  'fitpop_face_dprimes_v0': dpv0, 
                  'fitpop_face_dprimes_v3': dpv3,
                  'fitpop_face_dprimes_v6': dpv6,
                  'selectivities': face_selectivity_levels}

        R[('face_statistics', n)] = face_stats
    return R


def dprime(a, b):
    am = a.mean(0)
    bm = b.mean(0)
    av = a.var(0)
    bv = b.var(0)
    denom = np.sqrt((av + bv) / 2)
    return (am - bm) / denom


def get_face_dprimes(F, var_level='V0', inds=None, meta=None, cat='Faces'):
    if meta is None:
        meta = hvm_dataset.meta

    if inds is not None:
       meta = meta[inds]

    faces_inds = (meta['category'] == cat)
    nonfaces_inds = (meta['category'] != cat)

    if var_level is not None:
        faces_inds = faces_inds & (meta['var'] == var_level)
        nonfaces_inds = nonfaces_inds & (meta['var'] == var_level)

    print(cat, len(faces_inds), len(nonfaces_inds), F.shape)
    F_faces = F[faces_inds]
    F_nonfaces = F[nonfaces_inds]
    dprime_vals = dprime(F_faces, F_nonfaces)
    
    return dprime_vals


def get_human_cm(colname, query, tr_period, get_pred, get_actual,idataquery, conn, labelset):
    actuals = []
    preds = []
    for r in conn['mturk'][colname].find(query):
        resp = r['Response'][tr_period: ]
        imgData = r['ImgData'][tr_period:]
        if idataquery is not None:
            inds = [i for i in range(len(imgData)) if idataquery((resp[i], imgData[i]))]
            resp = [resp[i]  for i in inds]
            imgData = [imgData[i] for i in inds]
        pred = map(get_pred, zip(resp, imgData))
        actual = map(get_actual, zip(resp, imgData))
        actuals.extend(actual)
        preds.extend(pred)
    preds = np.array(preds)
    actuals = np.array(actuals)

    cm = classifier.get_confusion_matrix(actuals, preds, labelset)

    return cm
        

def get_facepackage_analysis(F, F_hvm):

    e = copy.deepcopy(reg_eval_it)
    e['metric_kwargs']['model_kwargs']['n_components'] = 25
    e['num_splits'] = 1
    e['test_q'].pop('var')
    e['npc_test'] = 1
    e['npc_train'] = 80
    resit = utils.compute_metric(F_hvm, hvm_dataset, e, attach_predictions=True, return_splits=True, attach_models=True)
    
    model = resit['split_results'][0]['model']
    popF = model.predict(F)
    pop_hvm = model.predict(F_hvm)

    import dldata.stimulus_sets.michael_cohen as mc
    dataset = mc.MichaelCohenFaceImagePackage()
    meta = dataset.meta
    dp = get_facepackage_dprimes(popF, pop_hvm, meta)
    dp1 = get_facepackage_dprimes(F, F_hvm, meta)
    
    return dp, dp1, popF, F


def get_face_dprimes_set(pop, var_level='V0', inds=None, meta=None, col='category'):
    if meta is None:
        meta = hvm_dataset.meta
    cats = np.unique(meta[col])
    R = {}
    for c in cats:
        R[c] = get_face_dprimes(pop, var_level=var_level, inds=inds, meta=meta, cat=c)
    return R


def get_facepackage_dprimes(pop, pop_hvm, meta):
    res = {}
    inds = meta['type'] == 'LocalizerImages'
    res['LocalizerImages'] = get_face_dprimes_set(pop[inds], var_level = None, inds=inds, meta=meta)
    inds = meta['type'] == 'TestImages'
    res['TestImagesWithNonNormal'] = get_face_dprimes_set(pop[inds], var_level = None, inds=inds, meta=meta)
    inds = (meta['type'] == 'TestImages') & (meta['subcategory'] == '')
    res['Testimages'] = get_face_dprimes_set(pop[inds], var_level = None, inds=inds, meta=meta)
    res['FacePackage_all'] = get_face_dprimes_set(pop, var_level = None, meta=meta)
    res['HvM_v0'] = get_face_dprimes_set(pop_hvm, var_level='V0')
    res['HvM_v3'] = get_face_dprimes_set(pop_hvm, var_level='V3')
    res['HvM_v6'] = get_face_dprimes_set(pop_hvm, var_level='V6')
    res['HvM_all'] = get_face_dprimes_set(pop_hvm, var_level=None)
    return res
