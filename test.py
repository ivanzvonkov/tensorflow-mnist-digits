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

    training_size = 2000
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

    # Testing features dict
    validation_features = load_features('t10k-images-idx3-ubyte.gz', testing_size, start=100)

    # Testing targets array
    validation_targets = load_labels('t10k-labels-idx1-ubyte.gz', testing_size, start=100)

    print 'Testing data imported'

    # Feature column for classifier, shape based on 28 by 28 pixel
    feature_columns = [tf.feature_column.numeric_column("image", shape=784)]

    # Training input function, returning iterator, shuffle automatically on
    training_input_fn = lambda: input_function(training_features, training_targets, batch_size=200)

    # Testing input fuction, returning iterator, shuffle automatically on
    testing_input_fn = lambda: input_function(testing_features, testing_targets, batch_size=200)

    # Prediction input function, one epoch
    prediction_input_fn_training = lambda: input_function(training_features, training_targets, num_epochs=1, shuffle=False)

    # Prediction input function, one epoch
    prediction_input_fn_testing = lambda: input_function(testing_features, testing_targets, num_epochs=1, shuffle=False)

    # Prediction input validation function, one epoch
    prediction_input_fn_validation = lambda: input_function(validation_features, validation_targets, num_epochs=1, shuffle=False)

    print 'Setting up classifier'
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=[10,20,10],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.007
        )
    )

    training_error = []
    testing_error = []

    # Loop for training
    for i in range(0, 10):
        print '------------------------'
        print 'RUN: ', i + 1
        print '------------------------'
        start_time = time.time()
        _ = dnn_classifier.train(
            input_fn=training_input_fn,
            steps=50,
        )
        end_time = time.time()
        print 'Training classifier: ', end_time - start_time

        # Calculate log loss
        training_predictions = list(dnn_classifier.predict(input_fn=prediction_input_fn_training)) # Array of prediction percentages
        training_probabilities = np.array([item['probabilities'] for item in training_predictions]) # 2d array of percentages of [0.043, ...]
        training_class_ids = np.array([item['class_ids'][0] for item in training_predictions]) # Array of prediction of 7
        training_pred_one_hot = tf.keras.utils.to_categorical(training_class_ids, 10) # 2d one hot array of [0. 0. ... 1. 0. 0.]

        testing_predictions = list(dnn_classifier.predict(input_fn=prediction_input_fn_testing))
        testing_probabilities = np.array([item['probabilities'] for item in testing_predictions])
        testing_class_ids = np.array([item['class_ids'][0] for item in testing_predictions])
        testing_pred_one_hot = tf.keras.utils.to_categorical(testing_class_ids, 10)

        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        testing_log_loss = metrics.log_loss(testing_targets, testing_pred_one_hot)

        training_error.append(training_log_loss)
        testing_error.append(testing_log_loss)

        print("%0.2f" % training_log_loss)
        print("%0.2f" % testing_log_loss)

    # Plot of training log loss vs testing log loss
    # Calculate final predictions (not probabilities, as above).
    testing_predictions = dnn_classifier.predict(input_fn=prediction_input_fn_testing)
    testing_predictions = np.array([item['class_ids'][0] for item in testing_predictions])
    testing_accuracy = metrics.accuracy_score(testing_targets, testing_predictions)
    print("Testing accuracy: %0.2f" % testing_accuracy)

    validation_predictions = dnn_classifier.predict(input_fn=prediction_input_fn_validation)
    validation_predictions = np.array([item['class_ids'][0] for item in validation_predictions])

    validation_accuracy = metrics.accuracy_score(validation_targets, validation_predictions)
    print("Validation accuracy: %0.2f" % validation_accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_error, label="training")
    plt.plot(testing_error, label="testing")
    plt.legend()
    plt.show()


    # # 2d array for holding accuracy
    # class_accuracy = np.full((10,10), 0).tolist()
    #
    # # Array for holding amount of times arrays are summed
    # class_sums = np.full(10, 0).tolist()
    #
    #
    # predictions = list(dnn_classifier.predict(input_fn=prediction_input_fn))
    #
    # metric_predictions = []
    #
    # for i, prediction in enumerate(predictions):
    #     #set up metric predictions
    #     metric_predictions.append( np.argmax(prediction['probabilities']) )
    #
    #     # Current class is the integer which we are getting predictions for
    #     current_class = int(prediction["classes"][0])
    #     # Sums array of accuracies with previous
    #     class_accuracy[current_class] = np.sum((class_accuracy[current_class], prediction["probabilities"]), axis=0)
    #     class_sums[current_class] += 1
    #
    # metric_predictions = np.array(metric_predictions, dtype=np.float64)
    # print metric_predictions.dtype
    # print metric_predictions.size
    # metric_predictions = tf.convert_to_tensor(metric_predictions)
    #
    # targets = np.array(testing_targets, dtype=np.float64)
    # print targets.dtype
    # print targets.size
    # targets = tf.convert_to_tensor(targets)
    # mean_squared_error = metrics.mean_squared_error(metric_predictions, targets)
    # print mean_squared_error
    #
    # for i in range(0, 10):
    #    if class_sums[i] != 0 :
    #     class_accuracy[i] = class_accuracy[i] / class_sums[i]
    #
    # fig, ax = plt.subplots()
    # im = ax.imshow(class_accuracy)
    #
    # # We want to show all ticks...
    # ax.set_xticks(np.arange(10))
    # ax.set_yticks(np.arange(10))
    #
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(np.arange(10))
    # ax.set_yticklabels(np.arange(10))
    #
    # # Loop over data dimensions and create text annotations.
    # for i in range(10):
    #     for j in range(10):
    #         text = ax.text(j, i, "{0:.2f}".format(class_accuracy[j][i]),
    #                        ha="center", va="center", color="w")
    #
    # ax.set_title("Accuracy of Numbers")
    # fig.tight_layout()
    #
    #
    # plt.show()
