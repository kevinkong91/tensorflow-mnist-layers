import helpers
import tensorflow as tf

mnist = helpers.read_mnist()

# NN configs
image_size = 28
labels_size = 10
hidden_size = 1024

# Define Input/Output layers
x_input = tf.placeholder(tf.float32, [None, image_size * image_size])
y_labels = tf.placeholder(tf.float32, [None, labels_size])

# Layers:
# - Input
# - Hidden (Dense with ReLU)
# - Output (Dense)

# Hidden layer
# activation parameter used to run ReLU activation function
# on the output of neurone
hidden = tf.layers.dense(
    inputs=x_input, units=hidden_size, activation=tf.nn.relu)

# Output dense layer
y_output = tf.layers.dense(inputs=hidden, units=labels_size)

# Define training
train_step, accuracy = helpers.build_training(y_labels, y_output)

# Run the training & test
helpers.train_test_model(mnist, x_input, y_labels, accuracy, train_step)
