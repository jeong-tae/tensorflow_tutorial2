# Tutorial for implementation of CNN with tensorflow
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .ops import *

class convNet(object):
    def __init__(self, config, sess, train_data, test_data):

        self.sess = sess

        self.train_data = train_data
        self.test_data = test_data
        self.train_range = config.train_range
        self.test_range = config.test_range

        # training detail
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.image_size = config.image_size
        self.channels = config.channels
        self.num_classes = config.num_classes
        self.max_epoch = config.max_epoch

        self.input_data = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.channels])
        self.input_labels = tf.placeholder(tf.float32, [None, self.num_classes])

    def build_model(self):
        """
        Write down your code here for what you want to make
        Find logits and optimize
        WANRNING: end point operation should be appended to op_list
        """

        op_list = []

        self.preds = tf.nn.softmax(logits) #
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.input_labels))
    
        op_list.append(optimize) #
        with tf.control_dependencies(op_list):
            self.op = tf.no_op(name='no_op')

    def train(self):
        total_preds = []
        total_labels = []
        total_loss = []
        
        for step, (data_in, label_in) in tqdm(enumerate(self.data_iteration(self.train_data, True)), desc = 'train'):
            _, loss, preds = self.sess.run([self.op, self.loss, self.preds],
                            feed_dict = {
                                    self.input_data: data_in,
                                    self.input_labels: label_in
                            })
            total_preds.append(preds)
            total_labels.append(label_in)
            total_loss.append(loss)

        total_loss = sum(total_loss)
        total_preds = np.concatenate(total_preds, axis = 0)
        total_labels = np.concatenate(total_labels, axis = 0)
        return total_loss, accuracy(total_preds, total_labels)
        
    def test(self):
        total_preds = []
        total_labels = []
        total_loss = []

        for step, (data_in, label_in) in tqdm(enumerate(self.data_iteration(self.test_data, False)), desc = 'test'):
            loss, preds = self.sess.run([self.loss, self.preds],
                            feed_dict = {
                                    self.input_data: data_in,
                                    self.input_labels: label_in
                            })
            total_preds.append(preds)
            total_labels.append(label_in)
            total_loss.append(loss)

        total_loss = sum(total_loss)
        total_preds = np.concatenate(total_preds, axis = 0)
        total_labels = np.concatenate(total_labels, axis = 0)
        return total_loss, accuracy(total_preds, total_labels)


    def run(self, task_name):
        tf.initialize_all_variables().run()

        for i in range(self.max_epoch):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()
            print("Epoch: %d, Train loss: %.3f, Train Acc: %.3f" % (i+1, train_loss, train_acc))
            print("Epoch: %d, Test loss: %.3f, Test Acc: %.3f" % (i+1, test_loss, test_acc))

        test_loss, test_acc = self.test()
        print("Task: %s, Test loss: %.3f, Test Acc: %.3f" % (str(task_name), test_loss, test_acc))    

    def demo(self):
        pass

    def data_iteration(self, data, is_train = True):
        """
        implement additional data preprocessing.
        this code is just before the data input

        """

        data_range = None
        if is_train:
            data_range = self.train_range
        else:
            data_range = self.test_range
        batch_len = len(data_range) // self.batch_size

        for l in xrange(batch_len):
            b_idx = data_range[self.batch_size * l:self.batch_size * (l+1)]
            
            batch_inputs = np.zeros((self.batch_size, self.image_size, self.image_size, self.channels), np.float32)
            batch_labels = np.zeros((self.batch_size), np.int32)

            for b in range(self.batch_size):
                image, label = np.copy(data[b_idx[b]])
                batch_inputs[b, :, :, :] = image
                batch_labels[b] = label

            batch_labels = (np.arange(self.num_classes) == batch_labels[:, None]).astype(np.float32)
            yield batch_inputs, batch_labels
