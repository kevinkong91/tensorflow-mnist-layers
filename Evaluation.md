# Evaluation of methods

## Single-layer Dense network
`src/dense.py`:

```
Step 0, training batch accuracy 0.1
Step 100, training batch accuracy 0.33
Step 200, training batch accuracy 0.57
Step 300, training batch accuracy 0.65
Step 400, training batch accuracy 0.67
Step 500, training batch accuracy 0.78
Step 600, training batch accuracy 0.8
Step 700, training batch accuracy 0.79
Step 800, training batch accuracy 0.84
Step 900, training batch accuracy 0.82
The end of training!
Test accuracy: 0.8405
Validation accuracy: 0.841
```

## Hidden layer dense network
`src/dense-relu.py`

```
Step 0, training batch accuracy 0.08
Step 100, training batch accuracy 0.92
Step 200, training batch accuracy 0.93
Step 300, training batch accuracy 0.9
Step 400, training batch accuracy 0.86
Step 500, training batch accuracy 0.88
Step 600, training batch accuracy 0.92
Step 700, training batch accuracy 0.96
Step 800, training batch accuracy 0.93
Step 900, training batch accuracy 0.97
The end of training!
Test accuracy: 0.9393
Validation accuracy: 0.9418
```

## Convolutional layer
`src/convolutional.py`

```
Step 0, training batch accuracy 0.05
Step 100, training batch accuracy 0.9
Step 200, training batch accuracy 0.93
Step 300, training batch accuracy 0.95
Step 400, training batch accuracy 0.95
Step 500, training batch accuracy 0.95
Step 600, training batch accuracy 0.98
Step 700, training batch accuracy 0.97
Step 800, training batch accuracy 0.97
Step 900, training batch accuracy 0.97
The end of training!
Test accuracy: 0.97
Validation accuracy: 0.9684
```