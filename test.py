import time

import tensorflow as tf
import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
from tensorflow.python.data import Dataset
from sklearn import metrics


# Converts header values
def convert_header(header, index):
    return (header[index + 2] * 256) + header[index + 3]


# Loads features from file and returns features dict
def load_features(filename, size):
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
        data = data[0:size]
        data = data / 255
        features = {"image": data}

    return features


# Loads labels from file and returns label array
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


# Input function used with dnn_classifier returns iterators of features, labels
def input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    feature, label = ds.make_one_shot_iterator().get_next()
    return feature, label



if __name__ == "__main__":

    training_size = 1000
    testing_size = 100

    # Training features dict
    training_features = load_features('train-images-idx3-ubyte.gz', training_size)

    # Training targets array
    training_targets = load_labels('train-labels-idx1-ubyte.gz', training_size)

    print 'Training data imported'

    # Testing features dict
    testing_features = load_features('t10k-images-idx3-ubyte.gz', testing_size)

    # Testing targets array
    testing_targets = load_labels('t10k-labels-idx1-ubyte.gz', testing_size)

    print 'Testing data imported'

    # Feature column for classifier, shape based on 28 by 28 pixel
    feature_columns = [tf.feature_column.numeric_column("image", shape=784)]

    # Training input function, returning iterator, shuffle automatically on
    training_input_fn = lambda: input_function(training_features, training_targets)

    # Testing input fuction, returning iterator, shuffle automatically on
    testing_input_fn = lambda: input_function(testing_features, testing_targets)

    # Prediction input function, one epoch
    prediction_input_fn = lambda: input_function(testing_features, testing_targets, num_epochs=1, shuffle=False)

    print 'Setting up classifier'
    # dnn_classifier = tf.estimator.DNNClassifier(
    #     feature_columns=feature_columns,
    #     n_classes=10,
    #     hidden_units=[10,10],
    #     optimizer=tf.train.ProximalAdagradOptimizer(
    #         learning_rate=0.005,
    #     )
    # )

    dnn_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.005,
        )
    )

    training_accuracy = []
    testing_accuracy = []

    # Loop for training
    for i in range(0, 2):
        print '------------------------'
        print 'RUN: ', i + 1
        print '------------------------'
        start_time = time.time()
        _ = dnn_classifier.train(
            input_fn=training_input_fn,
            steps=100,
        )
        end_time = time.time()
        print 'Training classifier: ', end_time - start_time

        result = dnn_classifier.evaluate(input_fn=training_input_fn, steps=50)
        print('Training set accuracy: {accuracy:0.3f}'.format(**result))
        training_accuracy = np.append(training_accuracy, result['accuracy'])

        result = dnn_classifier.evaluate(input_fn=testing_input_fn, steps=50)
        print('Training test accuracy: {accuracy:0.3f}'.format(**result))
        testing_accuracy = np.append(testing_accuracy, result['accuracy'])

    # Plot of training accuracy vs testing accuracy
    training_line = plt.plot(training_accuracy, label="Training")
    testing_line = plt.plot(testing_accuracy, label="Testing")
    plt.legend()

    # 2d array for holding accuracy
    class_accuracy = np.full((10,10), 0).tolist()

    # Array for holding amount of times arrays are summed
    class_sums = np.full(10, 0).tolist()


    predictions = list(dnn_classifier.predict(input_fn=prediction_input_fn))

    metric_predictions = []

    for i, prediction in enumerate(predictions):
        #set up metric predictions
        metric_predictions.append( np.argmax(prediction['probabilities']) )

        # Current class is the integer which we are getting predictions for
        current_class = int(prediction["classes"][0])
        # Sums array of accuracies with previous
        class_accuracy[current_class] = np.sum((class_accuracy[current_class], prediction["probabilities"]), axis=0)
        class_sums[current_class] += 1

    metric_predictions = np.array(metric_predictions, dtype=np.float64)
    print metric_predictions.dtype
    print metric_predictions.size
    metric_predictions = tf.convert_to_tensor(metric_predictions)

    targets = np.array(testing_targets, dtype=np.float64)
    print targets.dtype
    print targets.size
    targets = tf.convert_to_tensor(targets)
    mean_squared_error = metrics.mean_squared_error(metric_predictions, targets)
    print mean_squared_error

    for i in range(0, 10):
       if class_sums[i] != 0 :
        class_accuracy[i] = class_accuracy[i] / class_sums[i]

    fig, ax = plt.subplots()
    im = ax.imshow(class_accuracy)

    # We want to show all ticks...
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    # ... and label them with the respective list entries
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))

    # Loop over data dimensions and create text annotations.
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, "{0:.2f}".format(class_accuracy[j][i]),
                           ha="center", va="center", color="w")

    ax.set_title("Accuracy of Numbers")
    fig.tight_layout()


    plt.show()
