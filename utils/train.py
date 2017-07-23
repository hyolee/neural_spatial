import numpy as np
import h5py
import cPickle
import caffe
caffe_root = "/om/user/hyo/usr7/pkgs/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')
from optparse import OptionParser
import ast
from data import DataProvider
import time
import os
import subprocess

caffe.set_mode_gpu()
caffe.set_device(0)

def main(parser):
    # retrieve options
    (options, args) = parser.parse_args()
    net = options.net
    display = options.display
    base_lr = options.learning_rate
    lr_policy = options.lr_policy
    gamma = options.gamma
    stepsize = options.stepsize
    max_iter = options.max_iter
    momentum = options.momentum
    iter_size = options.iter_size
    weight_decay = options.weight_decay
    iter_snapshot = options.iter_resume
    snapshot = options.snapshot
    snapshot_prefix = options.snapshot_prefix
    test_interval = options.test_interval
    command_for_test = options.command_for_test
    weights = options.weights
    do_test = options.do_test
    dp_params = ast.literal_eval(options.dp_params)
    preproc = ast.literal_eval(options.preproc)

    # Data provider
    dp = DataProvider(dp_params, preproc)
    isize = dp.batch_size
    iter_size = dp.iter_size
    bsize = isize * iter_size

    # Load the net
    if weights is not None:
        print "Loading weights from " + weights
        N = caffe.Net(net, weights, caffe.TRAIN)
    else:
        if iter_snapshot==0:
            N = caffe.Net(net, caffe.TRAIN)
        else:
            print "Resuming training from " + snapshot_prefix + '_iter_' + str(iter_snapshot) + '.caffemodel'
            N = caffe.Net(net, snapshot_prefix + '_iter_' +  str(iter_snapshot) + '.caffemodel', caffe.TRAIN)

    # Save the net if training gets stopped
    import signal
    def sigint_handler(signal, frame):
        print 'Training paused...'
        print 'Snapshotting for iteration ' + str(iter)
        N.save(snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel')
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    # save weights before training
    if iter_snapshot == 0:
        print 'Snapshotting for initial weights'
        N.save(snapshot_prefix + '_iter_initial.caffemodel')

    # Start training
    loss = { key: 0 for key in N.outputs }
    prevs = {}
    for iter in range(iter_snapshot, max_iter):
        # clear param diffs
        for layer_name, lay in zip(N._layer_names, N.layers):
            for blobind, blob in enumerate(lay.blobs):
                blob.diff[:] = 0
        # clear loss  
        for k in loss.keys():
	    loss[k] = 0
       
        # update weights at every <iter_size> iterations 
        for i in range(iter_size):
	    # load data batch
            t0 = time.time()
            ind = iter * bsize + i * isize
            _data, _labels = dp.get_batch(ind)
            load_time = time.time() - t0

	    # set data as input
            N.blobs['data'].data[...] = _data
            for label_key in _labels.keys():
                N.blobs[label_key].data[...] = _labels[label_key].reshape(N.blobs[label_key].data.shape)

	    # Forward
#            t0 = time.time()
            out = N.forward()
#            forward_time = time.time()
	    # Backward
            N.backward()
#            backward_time = time.time() - forward_time
#            forward_time -= t0

            for k in out.keys():
                loss[k] += np.copy(out[k])

        # learning rate schedule
        if lr_policy == "step":
            learning_rate = base_lr * (gamma**(iter/stepsize))

        # print output loss
        print "Iteration", iter, "(lr: {0:.4f})".format( learning_rate )
        for k in np.sort(out.keys()):
            loss[k] /= iter_size
            print "Iteration", iter, ", ", k, "=", loss[k]
        sys.stdout.flush()

        # update filter parameters
#        t0 = time.time()
        for layer_name, lay in zip(N._layer_names, N.layers):
            for blobind, blob in enumerate(lay.blobs):
                diff = blob.diff[:]
                key = (layer_name, blobind)
                if key in prevs:
                    previous_change = prevs[key]
                else:
                    previous_change = 0.

		lr = learning_rate
		wd = weight_decay
#                if ("SpatialCorr" in lay.type):
#                    wd = 1000. * wd
		if (blobind == 1) and ("SpatialCorr" not in lay.type):
		    lr = 2. *lr
		    wd = 0.
		if lay.type == "BatchNorm":
		    lr = 0.
		    wd = 0.
                if ("SpatialCorr" in lay.type):
                    if "corronly" in net:
                        wd = 0.
                    else:
                        wd = 1.    # for training pos
		change = momentum * previous_change - lr * diff / iter_size - lr * wd * blob.data[:]

                blob.data[:] += change
		prevs[key] = change

#                # for debugging
#                if layer_name=='loss_scl':
#                    print "pos:", np.reshape(blob.data[:],-1)
#                    print "change:", np.reshape(change,-1)
#                    print "prev_change:", np.reshape(previous_change,-1)
#                    print "wd:", wd
#                    print "diff:", np.reshape(diff,-1)
#        update_time = time.time() - t0
#        print "loading: {0:.2f}, forward: {1:.2f}, backward: {2:.2f}, update: {3:.2f}".format(load_time, forward_time, backward_time, update_time)
        
        # save weights 
        if iter % snapshot == 0:
            print 'Snapshotting for iteration ' + str(iter)
            N.save(snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel')

        # test on validation set
        if do_test and (iter % test_interval == 0):
            print 'Test for iteration ' + str(iter)
            weights_test = snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel'
            if not os.path.exists(weights_test):
                N.save(snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel')
            command_for_test_iter = command_for_test + ' test.sh ' + net + ' ' + weights_test + ' "' + options.dp_params + '" "' + options.preproc + '"'
            print command_for_test_iter
            subprocess.call(command_for_test_iter, shell=True)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-n", "--net", dest="net")
    parser.add_option("-d", "--display", dest="display", default=1, type=int)
    parser.add_option("-l", "--base_lr", dest="learning_rate", type=float)
    parser.add_option("-c", "--lr_policy", dest="lr_policy", default=None)
    parser.add_option("-g", "--gamma", dest="gamma", default=None, type=float)
    parser.add_option("-z", "--stepsize", dest="stepsize", default=None, type=int)
    parser.add_option("-e", "--max_iter", dest="max_iter", type=int)
    parser.add_option("-m", "--momentum", dest="momentum", type=float)
    parser.add_option("-i", "--iter_size", dest="iter_size", default=1, type=int)
    parser.add_option("-w", "--weight_decay", dest="weight_decay", type=float)
    parser.add_option("-s", "--snapshot", dest="snapshot", default=500, type=int)
    parser.add_option("-p", "--snapshot_prefix", dest="snapshot_prefix", default=None)
    parser.add_option("-t", "--test-interval", dest="test_interval", default=1000, type=int)
    parser.add_option("--command-for-test", dest="command_for_test", default='sbatch --gres=gpu:1 --mem=5000')
    parser.add_option("-r", "--iter_resume", dest="iter_resume", default=0, type=int)
    parser.add_option("-f", "--weights", dest="weights", default=None)
    parser.add_option("--dp-params", dest="dp_params", default='', help="Data provider params")
    parser.add_option("--preproc", dest="preproc", default='', help="Data preprocessing params")
    parser.add_option("--do-test", dest="do_test", default=1, type=int)

    main(parser)
