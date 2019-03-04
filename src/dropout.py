import helpers
import tensorflow as tf

mnist = helpers.read_mnist()

image_size = 28
labels_size = 10
hidden_size = 1024

# Input layer - flattened images
x_input = tf.placeholder(tf.float32, [None, image_size*image_size])
y_labels = tf.placeholder(tf.float32, [None, labels_size])
should_drop = tf.placeholder(tf.bool)

# Reshape the image -
# to go back to the original structure of the image (2D representation)
x_image = tf.reshape(x_input, [-1, image_size, image_size, 1])


# Layers:
# - Input
# - Hidden1 (Conv with max pol)
# - Hidden2 (Conv with max pol)
# - Hidden3 (Dense with ReLU & Dropout)
# - Output (Dense)

# Hidden 1: Convolution with max pooling

conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=[
    5, 5], padding="same", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[
    5, 5], padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# For the next connection to Hidden #2, we must flatten the conv layer outputto 2D.
pool_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# Hidden 2
hidden = tf.layers.dense(
    inputs=pool_flat, units=hidden_size, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=hidden, rate=0.5, training=should_drop)

# Output dense layer
y_output = tf.layers.dense(inputs=hidden, units=labels_size)

# Define training
train_step, accuracy = helpers.build_training(y_labels, y_output)

# Run the training & test
helpers.train_test_model(mnist, x_input, y_labels, accuracy, train_step)
