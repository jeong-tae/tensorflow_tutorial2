# adjust hyper-parameter
import numpy as np

class base_config(object):
    def __init__(self, train_data, test_data):
        self.train_range = np.array(range(len(train_data)))
        self.test_range = np.array(range(len(test_data)))
        self.batch_size = 32
        self.lr = 0.05
        self.num_classes = 5
