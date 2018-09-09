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


if __name__ == "__main__":

    print 'Hello user!'

    one_feature = load_features('train-images-idx3-ubyte.gz', 1)

    targets = load_labels('train-labels-idx1-ubyte.gz', 1)

    # Feature column for classifier, shape based on 28 by 28 pixel
    feature_columns = [tf.feature_column.numeric_column("image", shape=784)]

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './mnist_saved_model/1536369603')
        predictor = tf.contrib.predictor.from_saved_model('./mnist_saved_model/1536369603')
        model_input = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=one_feature[0]))
        }))

        model_input = model_input.SerializeToString()
        output_dict = predictor({u'inputs': [model_input]})
        prediction_array = output_dict[u'scores'][0]
        prediction_list = list(prediction_array)
        prediction_value = prediction_list.index(max(prediction_list))
        prediction_accuracy = max(prediction_list)*100
        print 'I am guessing it\'s '  + str(prediction_value) + ' with '+ str("%.2f" % prediction_accuracy) + '% accuracy!'



