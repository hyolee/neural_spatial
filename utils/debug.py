import caffe
caffe_root = "/om/user/hyo/usr7/pkgs/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')
from data import DataProvider
import time

caffe.set_mode_gpu()
caffe.set_device(0)


class Net():
    def __init__(self, net, snapshot_prefix, dp_params, preproc, iter=None, test=False):
        if iter is not None:
            self.N = caffe.Net(net, snapshot_prefix + str(iter) + '.caffemodel', caffe.TRAIN)
            self.iter = iter
        else:
            if test:
                self.N = caffe.Net(net, snapshot_prefix, caffe.TEST)
#                self.N = caffe.Net(net, caffe.TRAIN)
            else:
                self.N = caffe.Net(net, snapshot_prefix, caffe.TRAIN)
            self.iter = 0
           
        # Data provider
        self.dp = DataProvider(dp_params, preproc)
        self.bsize = self.dp.batch_size

        self.prevs = {}
        self.test = test

    def forward(self):
        ind = self.iter * self.bsize 
        if self.test:
            _data, _labels = self.dp.get_batch_test(ind)
        else:
            _data, _labels = self.dp.get_batch(ind)

        # set data as input
        self.N.blobs['data'].data[...] = _data
        for label_key in _labels.keys():
            self.N.blobs[label_key].data[...] = _labels[label_key].reshape(self.N.blobs[label_key].data.shape)

        # Forward
#        t0 = time.time()
        out = self.N.forward()
        self.iter += 1
        return out

    def backward(self):
        self.N.backward()
        # update filter parameters
#        t0 = time.time()
        for layer_name, lay in zip(self.N._layer_names, self.N.layers):
            for blobind, blob in enumerate(lay.blobs):
                diff = blob.diff[:]
                key = (layer_name, blobind)
                if key in self.prevs:
                    previous_change = self.prevs[key]
                else:
                    previous_change = 0

                lr = 0.01
                wd = 0.0005
                momentum = 0.9
                if blobind == 1:
                    lr = 2 *lr
                    wd = 0
                if lay.type == "BatchNorm":
                    lr = 0
                    wd = 0
                change = momentum * previous_change - lr * diff  - lr * wd * blob.data[:]

                blob.data[:] += change
                self.prevs[key] = change
                
    def empty_diff(self):
        for layer_name, lay in zip(self.N._layer_names, self.N.layers):
            for blobind, blob in enumerate(lay.blobs):
                blob.diff[:] = 0 

#net = 'train_val_alexnet_scl_small.prototxt'
##snapshot_prefix = '/om/user/hyo/metrics/neural_spatial/test_loss/bvlc_alexnet.caffemodel'
##snapshot_prefix = 'snapshot/alexnet_inetFULL_scl_iter_'
#snapshot_prefix = 'snapshot/alexnet_scl_small_iter_'
#dp_params = {'data_path':'/om/user/yamins/.skdata/imagenet/ChallengeSynsets2013_offline_23d6ee636ade8ad3912204410d1acc23c10357cf/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/data.raw','data_key':'data','label_path':'/om/user/hyo/latent/googlenet2/data/label_inet.h5','label_key':['nc'],'cache_type':'hdf5','batch_size':256,'val_len':50000}
preproc = {'data_mean':'/om/user/hyo/caffe/imagenet_mean.npy','crop_size':227,'do_img_flip':True,'noise_level':10}
##N = Net(net, snapshot_prefix, dp_params, preproc, 1000)
