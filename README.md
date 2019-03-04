# Tensorflow Layers

## Exercises
- 1 Dense output layer
- All the above + 1 Hidden ReLU layer
- All the above + 1 Convolutional layer
- All the above + 1 Convolutional layer (2nd)
- All the above + Dropout function

See [results of the exercises](Evaluations.md)

## DROPOUT method

Previous methods took much longer due to overhead from the complexity of the network. In real-world applications, the number of steps for training should exceed 20K. The _dropout_ method can be applied to the hidden dense layer for improved performance.

Dropout either "shuts down" or "keeps" nodes with an explicit probability. It is used in the training phase and must be turned off for the evaluation phase.

To implement Dropout, a boolean placeholder should be created and the `tf.layer.dropout()` should be run on the hidden dense layer.
