import tensorflow as tf
import numpy as np

from config import base_config as config
from models import convNet
from flower_data import load_data

def run_task(data_dir, task_name = ''):

    train_data, test_data = load_data(data_dir)
    conv_config = config(train_data, test_data)

    with tf.Session() as sess:
        model = convNet(conv_config, sess, train_data, test_data)
        model.build_model()
        model.run(task_name)

def main(_):
    
    run_task('flowers/', 'flower_recognition')

if __name__ == '__main__':
    tf.app.run()



