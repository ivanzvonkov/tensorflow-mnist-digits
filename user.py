import tensorflow as tf
import os
from tensorflow.contrib import predictor
import tensorflow as tf
import gzip
import numpy as np

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
        features = {"image": data}

    return features


if __name__ == "__main__":

    print 'Hello user!'

    features = load_features('train-images-idx3-ubyte.gz', 1)

    predict_fn = predictor.from_saved_model('model/mnist-model/checkpoint')

    predictions = predict_fn(features)

    print predictions
