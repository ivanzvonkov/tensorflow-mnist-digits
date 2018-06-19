import time

import tensorflow as tf
import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
from tensorflow.python.data import Dataset
from sklearn import metrics

#converts header values
def convert_header(header, index):
    return (header[index+2] * 256) + header[index+3]

# returns features dict
def load_features(filename, size):
    with gzip.open(filename) as bytestream:

        #read header
        header = bytestream.read(16)
        header = np.frombuffer(header, dtype=np.uint8)
        num_images = convert_header(header, 4)
        IMAGE_SIZE = convert_header(header, 8)

        #read data
        data = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(data, dtype=np.uint8)
        data = data.reshape(num_images, IMAGE_SIZE*IMAGE_SIZE)
        data = np.array(data)
        data = data.astype(np.float32)
        data = data[0:size]
        data = data/255
        features = { "image": data}

    return features

# returns label array
def load_labels(filename, size):
    with gzip.open(filename) as bytestream:
        # read header
        header = bytestream.read(8)
        header = np.frombuffer(header, dtype=np.uint8)
        num_images = convert_header(header, 4)

        # read data
        data = bytestream.read(num_images)
        labels = np.frombuffer(data, dtype=np.uint8)
        labels = labels.astype(np.int32)
        labels = labels[0:size]

    return labels


# input function used with dnn_classifier
def input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

if __name__ == "__main__":

    training_features = load_features('train-images-idx3-ubyte.gz', 1000)
    training_targets = load_labels('train-labels-idx1-ubyte.gz', 1000)
    print 'Training data imported'

    testing_features = load_features('t10k-images-idx3-ubyte.gz', 100)
    testing_targets = load_labels('t10k-labels-idx1-ubyte.gz', 100)
    print 'Testing data imported'

    feature_columns = [tf.feature_column.numeric_column("image", shape=784)]

    training_input_fn = lambda: input_function(training_features, training_targets)
    testing_input_fn = lambda: input_function(testing_features, testing_targets)

    print 'Setting up classifier'
    # dnn_classifier = tf.estimator.DNNClassifier(
    #     feature_columns=feature_columns,
    #     n_classes=10,
    #     hidden_units=[10,10],
    #     optimizer=tf.train.ProximalAdagradOptimizer(
    #         learning_rate=0.001,
    #     )
    # )

    dnn_classifier = tf.estimator.LinearClassifier(
        feature_columns = feature_columns,
        n_classes=10,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.01,
        )
    )

    for i in range(0,10):
        print '------------------------'
        print 'RUN: ',i+1
        print '------------------------'
        start_time = time.time()
        _ = dnn_classifier.train(
            input_fn = training_input_fn,
            steps=50,
        )
        end_time = time.time()
        print 'Training classifier: ', end_time - start_time

        metrics = dnn_classifier.evaluate(input_fn= training_input_fn, steps=50)
        print('Training set accuracy: {accuracy:0.3f}'.format(**metrics))

        metrics = dnn_classifier.evaluate(input_fn=testing_input_fn, steps=50)
        print('Training test accuracy: {accuracy:0.3f}'.format(**metrics))





