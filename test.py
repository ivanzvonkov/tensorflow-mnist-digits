import tensorflow as tf
import numpy as np
import pandas as pd
import gzip

from tensorflow.python.data import Dataset

#converts header values
def convert_header(header, index):
    return (header[index+2] * 256) + header[index+3]

# returns features dict
def load_features(filename):
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
        data = data.astype(np.int32)
        features = { "image": data}

    return features

# returns label array
def load_labels(filename):
    with gzip.open(filename) as bytestream:
        # read header
        header = bytestream.read(8)
        header = np.frombuffer(header, dtype=np.uint8)
        num_images = convert_header(header, 4)

        # read data
        data = bytestream.read(num_images)
        labels = np.frombuffer(data, dtype=np.uint8)
        labels = labels.astype(np.int32)
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

def logger(text):
    print 'LOG:'+text

if __name__ == "__main__":

    training_features = load_features('train-images-idx3-ubyte.gz')
    training_targets = load_labels('train-labels-idx1-ubyte.gz')
    logger('Training data imported')

    testing_features = load_features('t10k-images-idx3-ubyte.gz')
    testing_targets = load_labels('t10k-labels-idx1-ubyte.gz')
    logger('Testing data imported')

    feature_columns = [tf.feature_column.numeric_column("image", shape=784)]

    logger('Setting up classifier')
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=[10,10],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
        )
    )

    logger('Training classifier')
    _ = dnn_classifier.train(
        input_fn = lambda: input_function(training_features, training_targets),
        steps=100
    )

    eval_result = dnn_classifier.evaluate(
        input_fn=lambda: input_function(testing_features, testing_targets))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))



