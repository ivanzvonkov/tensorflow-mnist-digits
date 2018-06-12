import tensorflow as tf
import numpy as np
import pandas as pd
import gzip

from tensorflow.python.data import Dataset

#converts header values
def convert_header(header, index):
    return (header[index+2] * 256) + header[index+3]

# returns features dict
def load_training_features(filename):
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
        #features = pd.Series({'images': data})
        #features = pd.DataFrame(features)
        features = { "image": np.array(data)}

    return features

# returns label array
def load_training_labels(filename):
    with gzip.open(filename) as bytestream:
        # read header
        header = bytestream.read(8)
        header = np.frombuffer(header, dtype=np.uint8)
        num_images = convert_header(header, 4)

        # read data
        data = bytestream.read(num_images)
        data = np.frombuffer(data, dtype=np.uint8)
        labels = np.array(data)
    return labels

# input function used with dnn_classifier
def input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    #features = {key: np.array(value) for key, value in dict(features).items()}

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
    # the program
    training_features = load_training_features('train-images-idx3-ubyte.gz')
    training_targets = load_training_labels('train-labels-idx1-ubyte.gz')
    print 'Training data imported'

    testing_features = load_training_features('t10k-images-idx3-ubyte.gz')
    testing

    feature_columns = [tf.feature_column.numeric_column("image", shape=784)]

    # Configure the linear regression model with our feature columns and optimizer.
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=[10,10],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
        )
    )

    _ = dnn_classifier.train(
        input_fn = lambda: input_function(features, targets),
        steps=100
    )

    prediction_input_fn =lambda: input_function(features, targets, num_epochs=1, shuffle=False)

    # Call predict() on the linear_regressor to make predictions.
    predictions = dnn_classifier.predict(input_fn=prediction_input_fn)

    # Format predictions as a NumPy array, so we can calculate error metrics.
    predictions = np.array([item['predictions'][0] for item in predictions])
    print predictions
    # Print Mean Squared Error and Root Mean Squared Error.
    #mean_squared_error = tf.metrics.mean_squared_error(predictions, targets)
    #root_mean_squared_error = tf.math.sqrt(mean_squared_error)
    #print "Mean Squared Error (on training data): %0.3f" % mean_squared_error
    #print "Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error


