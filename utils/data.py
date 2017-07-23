import h5py
import cPickle
import numpy as np
import time

class DataProvider:
    """
        dp_params: 
	  - 'data_path'
	  - 'data_key': for hdf5 file
          - 'label_path'
 	  - 'label_key': for hdf5 file
          - 'cache_type': for data and label files	
	  - 'batch_size': mini-batch size
	  - 'iter_size': (optional) # iterations of mini-batch per batch
	  - 'val_len': validataion set size at the end of the data file

	preproc:
	  - 'data_mean': path to data mean file (numpy)
	  - 'crop_size'
	  - 'do_img_flip'
	  - 'noise_level': (optional)
	  - 'resize_to': (optional) # TODO
    """

    def __init__(self, dp_params, preproc, init_iter=0):
        # retrieve data and label
        self.cache_type = dp_params['cache_type']
        if (self.cache_type=='hdf5'):
            F_DATA = h5py.File(dp_params['data_path'], mode='r')
            self.data = F_DATA[dp_params['data_key']]
            self.labels = h5py.File(dp_params['label_path'], mode='r')
            self.label_keys = dp_params['label_key']

        self.train_len = self.data.shape[0] - dp_params['val_len']    
        self.val_len = dp_params['val_len']

        # data preprocessing params
        self.data_mean = np.load(preproc['data_mean'])
        self.crop_size = preproc['crop_size']
        self.do_img_flip = preproc['do_img_flip']
        if 'noise_level' in preproc.keys():
            self.noise_level = preproc['noise_level']
        else:
            self.noise_level = False
        if 'resize_to' in preproc.keys():
            self.resize_to = preproc['resize_to']
        else:
            self.resize_to = (3, 256, 256)
    
        # training-related params
        self.batch_size = dp_params['batch_size']
        if 'iter_size' in dp_params.keys():
            self.iter_size = dp_params['iter_size']
        else:
            self.iter_size = 1

    def get_batch(self, ind):
        _labels = {}
        perm = np.random.RandomState().permutation(self.batch_size)
        # load data batch
        start = ind % self.train_len
        end = (ind + self.batch_size) % self.train_len
        if (start > end):
            _data = np.concatenate((self.data[start:self.train_len], self.data[:end]), axis=0)[perm]
            for k in self.label_keys:
                _labels[k] = np.concatenate((self.labels[k][start:self.train_len], self.labels[k][:end]), axis=0)[perm]
        else:
            _data = self.data[start:end][perm]
            for k in self.label_keys:
                _labels[k] = self.labels[k][start:end][perm]

        # data preprocessing 
        _data = _data.reshape((_data.shape[0], self.resize_to[0], self.resize_to[1], self.resize_to[2])) 
        # mean subtraction
        _data = _data - self.data_mean
        # data augmentation
        _data_cropped = np.empty([_data.shape[0], _data.shape[1], self.crop_size, self.crop_size], dtype=np.float32)
        crop_max_start = _data.shape[2] - self.crop_size
        for i in range(_data.shape[0]):
        #    t0 = time.time()
            # random crop
            startY, startX = np.random.randint(0,crop_max_start), np.random.randint(0, crop_max_start)
            endY, endX = startY + self.crop_size, startX + self.crop_size
            pic = _data[i, :, startY:endY, startX:endX]
        #    t1 = time.time()
            # flip image with 50% chance
            if self.do_img_flip and np.random.randint(2)==0:
                pic = pic[:, :, ::-1]
        #    t2 = time.time()
            # add noise
            if self.noise_level and np.random.randint(2)==0:
                noise_coefficient = np.random.uniform(low=0, high=self.noise_level)
                ns = (np.random.RandomState().randn(*pic.shape) * noise_coefficient).astype(pic.dtype)
                mx = np.abs(pic).max()
                pic += ns
                pic *= mx / np.abs(pic).max()
            _data_cropped[i, :, :, :] = pic
        #    t3 = time.time()
        #    print "crop: {0:.2f}, flip: {1:.2f}, noise: {2:.2f}".format(t1-t0, t2-t1, t3-t2)
        return _data_cropped, _labels

    def get_batch_test(self, ind):
        _labels = {}
     #   perm = np.random.RandomState().permutation(self.batch_size)
        # load data batch
        start = ind % self.val_len + self.train_len
        end = (ind + self.batch_size) % self.val_len + self.train_len
        if (start > end):
            _data = np.concatenate((self.data[start:], self.data[self.train_len:end]), axis=0)#[perm]
            for k in self.label_keys:
                _labels[k] = np.concatenate((self.labels[k][start:], self.labels[k][self.train_len:end]), axis=0)#[perm]
        else:
            _data = self.data[start:end]#[perm]
            for k in self.label_keys:
                _labels[k] = self.labels[k][start:end]#[perm]

        # data preprocessing 
        _data = _data.reshape((_data.shape[0], self.resize_to[0], self.resize_to[1], self.resize_to[2])) 
        # mean subtraction
        _data = _data - self.data_mean
        # center crop
        crop_start = (_data.shape[2] - self.crop_size) / 2
        _data_cropped = _data[:, :, crop_start:(crop_start+self.crop_size), crop_start:(crop_start+self.crop_size)]
        return _data_cropped, _labels

# example code
if __name__ == "__main__":
    dp_params = {'data_path': '/om/user/yamins/.skdata/imagenet/ChallengeSynsets2013_offline_23d6ee636ade8ad3912204410d1acc23c10357cf/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/data.raw',
                    'data_key': 'data',
                    'label_path': '/om/user/hyo/latent/googlenet2/data/label_inet.h5',
                    'label_key': ['nc'],
                    'cache_type': 'hdf5',
                    'batch_size': 256,
                    'iter_size': 1,
                    'val_len': 50000}
    preproc = {'data_mean': '/om/user/hyo/caffe/imagenet_mean.npy',
                    'crop_size': 227,
                    'do_img_flip': True,
                    'noise_level': 10}
    dp = DataProvider(dp_params, preproc)
