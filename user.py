import time
import tensorflow as tf
import numpy as np
import gzip
import matplotlib.pyplot as plt
from tensorflow.python.data import Dataset
from sklearn import metrics
import os

# Converts header values
def convert_header(header, index):
    return (header[index + 2] * 256) + header[index + 3]

# Loads features from file and returns features dict
def load_features(filename, size, start=0,):
    with gzip.open(filename) as bytestream:
        # Read header
        header = bytestream.read(16)
        header = np.frombuffer(header, dtype=np.uint8)
        num_images = convert_header(header, 4)
        IMAGE_SIZE = convert_header(header, 8)

        # Read data
        data = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(data, dtype=np.uint8)
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)
        data = np.array(data)
        data = data.astype(np.float32)
        data = data[start:start+size]
        data = data / 255
        #features = {"image": data}
        features = data

    return features

# Loads labels from file and returns label array
def load_labels(filename, size, start=0):
    with gzip.open(filename) as bytestream:
        # read header
        header = bytestream.read(8)
        header = np.frombuffer(header, dtype=np.uint8)
        num_images = convert_header(header, 4)

        # read data
        data = bytestream.read(num_images)
        labels = np.frombuffer(data, dtype=np.uint8)
        labels = labels.astype(np.int32)
        labels = labels[start:start+size]

    return labels

# Input function used with dnn_classifier returns iterators of features, labels
def input_function(features, targets=None, batch_size=1, shuffle=True, num_epochs=None):
    # Construct a dataset, and configure batching/repeating.

    if targets is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, targets)

        # Convert the inputs to a Dataset.

    ds = Dataset.from_tensor_slices(inputs)

    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    if targets is None:
        feature = ds.make_one_shot_iterator().get_next()
        return feature
    else:
        feature, label = ds.make_one_shot_iterator().get_next()
        return feature, label


if __name__ == "__main__":

    print 'Hello user!'

    features = load_features('train-images-idx3-ubyte.gz', 1)

    targets = load_labels('train-labels-idx1-ubyte.gz', 1)

    predictions_input_fn = lambda: input_function(features, targets)

    # Feature column for classifier, shape based on 28 by 28 pixel
    feature_columns = [tf.feature_column.numeric_column("image", shape=784)]

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './mnist_saved_model/1536369603')
        predictor = tf.contrib.predictor.from_saved_model('./mnist_saved_model/1536369603')
        model_input = tf.train.Example(features=tf.train.Features(feature={"image": tf.train.Feature(int64_list=tf.train.Int64List(value=features[0]))}))
        model_input = model_input.SerializeToString()
        output_dict = predictor({"predictor_inputs": [model_input]})
        y_predicted = output_dict["pred_output_classes"][0]


