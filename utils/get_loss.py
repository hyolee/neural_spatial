import numpy as np
import sys

def append_array(it_p, t_p, it, t):
    if len(it_p)==0:
        return it, t
    ind = np.where(it_p == it[0])[0]
    if len(ind)==0:
        return np.append(it_p, it), np.append(t_p, t)
    ind = ind[0]
    return np.append(it_p[:ind], it), np.append(t_p[:ind], t)

files = sys.argv[1:]
cat = np.array([])
scl = np.array([])
it_cat = np.array([])
it_scl = np.array([])
for file in files:
    cat_here = np.array([])
    scl_here = np.array([])
    it_scl_here = np.array([])
    it_cat_here = np.array([])
    with open(file) as f:
        for line in f:
            if 'Iteration' in line and 'lr' not in line:
                it = int( (line.split('Iteration ')[1]).split(',')[0] )
                if 'loss_scl' in line:
                    it_scl_here = np.append(it_scl_here, it)
                    scl_here = np.append(scl_here, float( (line.split('= ')[1]).split(' (')[0] ))
#                    scl_here = np.append(scl_here, float( (line.split('[ ')[1]).split(']')[0] ))
                elif 'loss' in line:
                    it_cat_here = np.append(it_cat_here, it)
                    cat_here = np.append(cat_here, float( (line.split('= ')[1]).split(' (')[0] ))
    it_cat, cat = append_array(it_cat, cat, it_cat_here, cat_here)
    if len(it_scl_here) > 0:
        it_scl, scl = append_array(it_scl, scl, it_scl_here, scl_here)
temp = {'cat': cat, 'scl': scl, 'iter_scl': it_scl, 'iter_cat': it_cat}

import os
import cPickle
if '_r' in file:
    var = file.split('/')[-1].split('_r')[0]
else:
    var = file.split('/')[-1].split('.')[0]
    print file
    print  var
filename = '/'.join(file.split('/')[:-1]) + '/accuracy_' + var + '.pkl'
print filename
if os.path.exists(filename):
    dic = cPickle.load(open(filename))
    dic['iter_cat'], dic['cat'] = append_array(dic['iter_cat'], dic['cat'], it_cat, cat)
    if len(it_scl) > 0:
        dic['iter_scl'], dic['scl'] = append_array(dic['iter_scl'], dic['scl'], it_scl, scl)
else:
    dic = temp
with open(filename, 'wb') as f:
    cPickle.dump(dic, f)
