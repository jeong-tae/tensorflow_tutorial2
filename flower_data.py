import glob
import random
import cPickle
from scipy import ndimage, misc
import os

label_dict = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4
}

WIDTH = 224
HEIGHT = 224

def image_label(file_name):
    for name in label_dict.keys():
        if file_name in name:
            return label_dict[name]

    raise "unknown label %s" % file_name

def load_data(data_dir = 'flowers/'):
    train_path = data_dir + 'train.pkl'
    test_path = data_dir + 'test.pkl'

    train_data = None
    test_data = None
    if os.path.exists(train_path):
        train_data = load_pkl(train_path)
        test_data = load_pkl(test_path)
    else:
        train_data, test_data = data2pickle(data_dir)

    return train_data, test_data

def data2pickle(data_dir = 'flowers/'):

    train_files = glob.glob('%s/raw-data/train/*/*.jpg' % data_dir)
    test_files = glob.glob('%s/raw-data/validation/*/*.jpg' % data_dir)

    random.shuffle(train_files)

    train_data = []
    for f in train_files:
        im = ndimage.imread(f)
        k = misc.imresize(im, (WIDTH, HEIGHT))
        v = image_label(f)
        train_data.append((k, v))

    test_data = []
    for f in test_files:
        im = ndimage.imread(f)
        k = misc.imresize(im, (WIDTH, HEIGHT))
        v = image_label(f)
        test_data.append((k, v))
    
    train_path = data_dir + 'train.pkl'
    test_path = data_dir + 'test.pkl'

    save_pkl(train_path, train_data)
    save_pkl(test_path, test_data)

    return train_data, test_data
