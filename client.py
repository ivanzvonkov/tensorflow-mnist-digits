import tensorflow as tf
import numpy as np
import gzip
import random
from PIL import Image

# Converts header values
from paint import Paint


class Client:
    def convert_header(self, header, index):
        return (header[index + 2] * 256) + header[index + 3]

    # Loads features from file and returns features dict
    def load_features(self, filename, size, start=0,):
        with gzip.open(filename) as bytestream:
            # Read header
            header = bytestream.read(16)
            header = np.frombuffer(header, dtype=np.uint8)
            num_images = self.convert_header(header, 4)
            IMAGE_SIZE = self.convert_header(header, 8)

            # Read data
            data = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
            data = np.frombuffer(data, dtype=np.uint8)
            data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)
            data = np.array(data)
            data = data.astype(np.float32)
            data = data[start:start+size]
            data = data / 255
            features = data

        return features

    # Loads labels from file and returns label array
    def load_labels(self, filename, size, start=0):
        with gzip.open(filename) as bytestream:
            # read header
            header = bytestream.read(8)
            header = np.frombuffer(header, dtype=np.uint8)
            num_images = self.convert_header(header, 4)

            # read data
            data = bytestream.read(num_images)
            labels = np.frombuffer(data, dtype=np.uint8)
            labels = labels.astype(np.int32)
            labels = labels[start:start+size]

        return labels

    # Predict one 2d array feature
    def predict_one(self, one_feature):

        model_input = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=one_feature))
        }))

        model_input = model_input.SerializeToString()

        output_dict = self.predictor({u'inputs': [model_input]})
        prediction_list = list(output_dict[u'scores'][0])
        prediction_value = prediction_list.index(max(prediction_list))
        prediction_accuracy = max(prediction_list) * 100

        return prediction_value, prediction_accuracy

    # Predict random digit from data
    def predict_random_digit(self):
        while (True):
            random_number = random.randint(0, self.size)
            prediction_value, prediction_accuracy = self.predict_one(self.features[random_number])
            self.drawing_from_array(self.features[random_number])
            print 'Guessing it\'s ' + str(prediction_value) + ' with ' + str("%.2f" % prediction_accuracy) + '% accuracy.'
            print 'Answer is ' + str(self.targets[random_number])
            user_input = raw_input("Quit (q) or Continue (Enter)")
            if user_input == 'q':
                break

    # Predicts drawn digit
    def predict_drawn_digit(self, drawn_feature):
        drawn_feature = np.array(drawn_feature)
        drawn_feature = drawn_feature.flatten()
        prediction_value, prediction_accuracy = self.predict_one(drawn_feature)
        print 'Guessing it\'s ' + str(prediction_value) + ' with ' + str("%.2f" % prediction_accuracy) + '% accuracy.'

    # Starts paint object
    def start_paint(self):
        observer = lambda drawn_feature: self.predict_drawn_digit(drawn_feature)
        Paint(observer)

    # Creates image from array for testing
    def drawing_from_array(self, features):
        data = np.zeros((28*28, 3), dtype=np.uint8)

        for i, feature in enumerate(features):
            color = 255*feature
            data[i] = np.array([color, color, color])

        data = np.reshape(data, (28,28,3))

        img = Image.fromarray(data, 'RGB')
        img = img.resize((150,150))
        img.show()

    def __init__(self):

        print 'Hello'

        self.size = 1000

        self.features = self.load_features('train-images-idx3-ubyte.gz', self.size, start=9000)

        self.targets = self.load_labels('train-labels-idx1-ubyte.gz', self.size, start=9000)

        # Feature column for classifier, shape based on 28 by 28 pixel
        feature_columns = [tf.feature_column.numeric_column("image", shape=784)]

        with tf.Session() as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './mnist_saved_model/1536369603')
            self.predictor = tf.contrib.predictor.from_saved_model('./mnist_saved_model/1536369603')

            #self.predict_random_digit()

            self.start_paint()



Client()