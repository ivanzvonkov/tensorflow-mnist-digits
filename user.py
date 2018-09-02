import tensorflow as tf

if __name__ == "__main__":
    # build your model (same as training)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, 'model/model.ckpt')

    sess.run(y_pred, feed_dict={x: input_data})